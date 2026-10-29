
import os
import sys
import json
from datetime import datetime

# Force l'encodage UTF-8 sur la console Windows pour afficher correctement
# les caractères générés par le LLM (tirets insécables, guillemets, etc.)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from execution.agent import Agent
from execution.masRunner import MASRunner
from execution.traceLogger import TraceLogger

from explanation.rollback import Rollback
from explanation.alternatives import AlternativeGenerator
from explanation.counterfactual import CounterfactualSimulator
from explanation.causeIdentifier import CauseIdentifier
from explanation.explanation_gen import ExplanationGenerator

from evaluation.llmJudge import LLMJudge
from evaluation.metrics import FidelityMetrics

from config import OUTPUT_EXPLANATIONS, OUTPUT_TRACES, OUTPUT_GRAPHS

for folder in [OUTPUT_TRACES, OUTPUT_EXPLANATIONS, OUTPUT_GRAPHS]:
    os.makedirs(folder, exist_ok=True)

def run_phase1() -> tuple[str, dict]:
    # exécution réelle de la négociation

    agent_a = Agent(
        name="Vendeur",
        role="Vendeur d'un smartphone haut de gamme",
        objective="Vendre le produit à au moins 800€. Prix initial : 1000€."
    )
    agent_b = Agent(
        name="Acheteur",
        role="Acheteur avec un budget serré",
        objective="Acheter le smartphone à moins de 750€. Budget max : 850€."
    )
    agent_c = Agent(
        name="Mediateur",
        role="Médiateur neutre",
        objective="Aider à trouver un compromis équitable entre l'acheteur et le vendeur."
    )
    agent_d = Agent(
        name="Expert",
        role="Expert technique indépendant",
        objective="Évaluer la valeur réelle du smartphone en fonction de ses caractéristiques et informer les parties."
    )
    agent_e = Agent(
        name="Concurrent",
        role="Second acheteur potentiel",
        objective="Acheter le smartphone si le prix tombe à 800€. Tu es prêt à surenchérir sur le premier acheteur si nécessaire."
    )
    agent_f = Agent(
        name="Observateur",
        role="Observateur du marché",
        objective="Analyser les tendances de prix et donner des conseils sur le moment opportun pour conclure la vente."
    )

  
    runner = MASRunner(agents=[agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], scenario="S2_smartphone_6agents")
    
    trace_path = runner.run(
        initial_message="Bonjour, je vends ce smartphone neuf pour 1000€. Qui est intéressé ?"
    )

    trace = TraceLogger.load(trace_path)
    return trace_path, trace

def run_phase2(trace: dict) -> tuple[dict, dict, list, dict]:
    # génération de l'explication CEMA

    # cible la première décision de l'acheteur 
    target_agent = "Acheteur"
    target_timestamp = 1

    #premier pas de l'acheteur dans la trace
    buyer_steps = [s for s in trace["steps"] if s["agent"] == target_agent]
    if buyer_steps:
        target_timestamp = buyer_steps[0]["timestamp"]
    
    print(f"\n  [Cible] Agent: {target_agent} | t={target_timestamp}")

    # rollback
    rollback = Rollback(trace)
    context = rollback.extract_context(target_agent, target_timestamp)

    #alternatives
    alt_gen = AlternativeGenerator()
    alternatives = alt_gen.generate(context, rollback)

    #simulation contrefactuelle
    cf_sim = CounterfactualSimulator()
    cf_traces = cf_sim.simulate_all(context, alternatives, rollback)

    #identification des causes
    cause_id = CauseIdentifier()
    causes = cause_id.identify(context, cf_traces)

    exp_gen = ExplanationGenerator()
    explanation = exp_gen.generate(
        context=context,
        causes=causes,
        cf_traces=cf_traces,
        session_id=trace["session_id"]
    )

    return context, causes, cf_traces, explanation

def run_phase3(trace: dict, context: dict, causes: dict, explanation: dict) -> dict:
    #evaluation de la fidélité

    judge = LLMJudge()
    rollback = Rollback(trace)
    exp_gen = ExplanationGenerator()

    # ci
    explanations_ci = judge.evaluate_consistency(context, rollback, exp_gen)

    # jugement LLM
    judge_result = judge.evaluate(
        trace=trace,
        explanation=explanation["text_explanation"],
        target_agent=context["target_agent"],
        target_timestamp=context["target_timestamp"]
    )

    # calcul des métriques
    metrics = FidelityMetrics.compute_all(
        explanations_ci=explanations_ci,
        explanation=explanation["text_explanation"],
        causes=causes,
        ve_score=judge_result["ve_score"]
    )

    print(f"\n  [score global] : {metrics['global_score']}/1.0")
    print(f"  [interprétation] : {metrics['interpretation']}")

    return metrics

def main():
    try:
        trace_path, trace = run_phase1()

        context, causes, cf_traces, explanation = run_phase2(trace)
        metrics = run_phase3(trace, context, causes, explanation)

        print(f"  ID Session : {trace['session_id']}")
        print(f"  Graphe     : outputs/graphs/graph_{trace['session_id']}.png")

    except Exception as e:
        print(f"\n  [ERREUR] erreur  : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
