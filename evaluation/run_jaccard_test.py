import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.agent import Agent
from execution.masRunner import MASRunner
from execution.traceLogger import TraceLogger
from explanation.rollback import Rollback
from explanation.alternatives import AlternativeGenerator
from explanation.counterfactual import CounterfactualSimulator
from explanation.causeIdentifier import CauseIdentifier


def run_controlled_session() -> tuple:

    #buyer :makes price proposals
    agent_acheteur = Agent(
        name="Acheteur",
        role="Acheteur intéressé par un smartphone",
        objective="Acheter le smartphone à moins de 700€. Budget maximum : 750€."
    )

    #seller reacts to price proposals
    agent_vendeur = Agent(
        name="Vendeur",
        role="Vendeur du smartphone",
        objective="Vendre le smartphone à au moins 850€. Ne pas descendre sous 800€."
    )

    #mediator: reacts to negotiation dynamics
    agent_mediateur = Agent(
        name="Mediateur",
        role="Médiateur de la négociation",
        objective="Aider à trouver un compromis équitable entre 750€ et 850€."
    )

    #tchnical spec : gives the same fixed specs
    agent_technic = Agent(
        name="AgentTechnic",
        role="Agent de documentation technique",
        objective=(
            "Ton UNIQUE rôle est de fournir les spécifications techniques du smartphone. "
            "Tu dois TOUJOURS donner exactement les mêmes informations techniques : "
            "Écran 6.1 pouces OLED, RAM 6 Go, Stockage 128 Go, Appareil photo 12 MP, "
            "Batterie 3500 mAh, Poids 174g, OS Android 14, Processeur Snapdragon 888. "
            "Tu NE DOIS PAS commenter le prix, les négociations, les offres ou les compromis. "
            "Ignore toute mention de prix. Réponds UNIQUEMENT avec les specs techniques ci-dessus."
        )
    )

    runner = MASRunner(
        agents=[agent_acheteur, agent_vendeur, agent_mediateur, agent_technic],
        scenario="S_jaccard_selectivity_test"
    )
    trace_path = runner.run(
        initial_message=(
            "Bonjour, je voudrais acheter ce smartphone mais je trouve que 1000€ "
            "c'est beaucoup trop cher. Je propose 650€. Le vendeur accepte-t-il ?"
        )
    )
    trace = TraceLogger.load(trace_path)
    print(f"  Session: {trace['session_id']} | Steps: {len(trace['steps'])}")
    return trace_path, trace

#compute mean pairwise Jaccard similarity for each agent's reactions
def compute_jaccard_per_agent(cf_traces: list, other_agents: list) -> dict:
    def tokenize(t):
        return set(re.findall(r'\b\w+\b', t.lower()))

    results = {}
    for agent in other_agents:
        reactions = [
            cf.get("reactions", {}).get(agent, "")
            for cf in cf_traces
            if cf.get("reactions", {}).get(agent, "")
        ]
        if len(reactions) < 2:
            results[agent] = {"avg_similarity": None, "influenced": None,
                              "reason": "Not enough data"}
            continue

        tokens = [tokenize(r) for r in reactions]
        sims = []
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                inter = len(tokens[i] & tokens[j])
                union = len(tokens[i] | tokens[j])
                sims.append(inter / union if union > 0 else 0)

        avg_sim = round(sum(sims) / len(sims), 3) if sims else 1.0
        influenced = avg_sim < 0.5

        results[agent] = {
            "avg_similarity": avg_sim,
            "influenced": influenced,
            "reason": (
                f"Jaccard={avg_sim:.3f} < 0.5 → reactions vary → INFLUENCED"
                if influenced else
                f"Jaccard={avg_sim:.3f} ≥ 0.5 → reactions stable → NOT influenced"
            )
        }
    return results


def run_jaccard_test():
    trace_path, trace = run_controlled_session()

    steps = trace["steps"]
    target_agent = steps[0]["agent"] if steps else "Vendeur"
    target_timestamp = steps[0]["timestamp"] if steps else 0

    print(f"\n  Target: {target_agent} @ t={target_timestamp}")

    #run SCRIBE pipeline
    rollback = Rollback(trace)
    context  = rollback.extract_context(target_agent, target_timestamp)

    alt_gen      = AlternativeGenerator()
    alternatives = alt_gen.generate(context, rollback)

    cf_sim    = CounterfactualSimulator()
    cf_traces = cf_sim.simulate_all(context, alternatives, rollback)

    cause_id = CauseIdentifier()
    causes   = cause_id.identify(context, cf_traces)

    #Jaccard per agent
    other_agents = context["other_agents"]
    jaccard_details = compute_jaccard_per_agent(cf_traces, other_agents)

    print(f"  Target agent : {target_agent} @ t={target_timestamp}")
    print(f"  Agents influenced (SCRIBE): {causes['agents_influenced']}")
    print()

    for agent, data in jaccard_details.items():
        status = "INFLUENCED ✓" if data["influenced"] else "NOT influenced ✗"
        expected = "passive" if agent == "AgentTechnic" else "active"
        correct = (
            (agent == "AgentTechnic" and not data["influenced"]) or
            (agent != "AgentTechnic" and data["influenced"])
        )
        flag = " CORRECT" if correct else " UNEXPECTED"
        print(f"  {agent:<20} | Jaccard={data['avg_similarity']} | {status} | [{expected}] {flag}")

    print()
    print("  Threshold selectivity: VALIDATED" if any(
        not v["influenced"] for v in jaccard_details.values() if v["influenced"] is not None
    ) else "  all agents still flagged as influenced — threshold not discriminating")

    os.makedirs("outputs", exist_ok=True)
    result = {
        "session_id": trace["session_id"],
        "target_agent": target_agent,
        "target_timestamp": target_timestamp,
        "agents_influenced_scribe": causes["agents_influenced"],
        "jaccard_per_agent": jaccard_details,
        "threshold": 0.5,
    }
    with open("outputs/jaccard_test_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    #text report
    lines = [
        "Jaccard Selectivity Test Report",
        f"Target: {target_agent} @ t={target_timestamp}",
        f"SCRIBE influenced: {causes['agents_influenced']}",
    ]
    for agent, data in jaccard_details.items():
        lines.append(
            f"{agent:<20} Jaccard={data['avg_similarity']}  "
            f"{'INFLUENCED' if data['influenced'] else 'NOT influenced'}"
        )
    with open("outputs/jaccard_test_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n  saved: outputs/jaccard_test_result.json")
    print("  saved: outputs/jaccard_test_report.txt")
    return result


if __name__ == "__main__":
    run_jaccard_test()
