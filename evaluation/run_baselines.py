#runs SCRIBE with 2 baselines on the same trace 
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from evaluation.baseline_textual import TextualBaseline
from evaluation.baseline_temporal import TemporalBaseline


#run the MAS once
def run_mas() -> tuple:

    agent_a = Agent(name="Vendeur",    role="Vendeur d'un smartphone haut de gamme",
                    objective="Vendre le produit à au moins 800€. Prix initial : 1000€.")
    agent_b = Agent(name="Acheteur",   role="Acheteur avec un budget serré",
                    objective="Acheter le smartphone à moins de 750€. Budget max : 850€.")
    agent_c = Agent(name="Mediateur",  role="Médiateur neutre",
                    objective="Aider à trouver un compromis équitable.")
    agent_d = Agent(name="Expert",     role="Expert technique indépendant",
                    objective="Évaluer la valeur réelle du smartphone.")
    agent_e = Agent(name="Concurrent", role="Second acheteur potentiel",
                    objective="Acheter si le prix tombe à 800€.")
    agent_f = Agent(name="Observateur",role="Observateur du marché",
                    objective="Analyser les tendances de prix.")

    runner = MASRunner(
        agents=[agent_a, agent_b, agent_c, agent_d, agent_e, agent_f],
        scenario="S2_baseline_comparison"
    )
    trace_path = runner.run(
        initial_message="Bonjour, je vends ce smartphone neuf pour 1000€. Qui est intéressé ?"
    )
    trace = TraceLogger.load(trace_path)
    print(f"  Trace loaded: {trace['session_id']}")
    return trace_path, trace


#evaluate one explanation with metrics
def evaluate(trace, context, causes, explanation_text, session_tag) -> dict:
    judge  = LLMJudge()
    exp_gen = ExplanationGenerator()
    rollback = Rollback(trace)

    # CI,run N=3 repetitions
    print(f"\n  [CI] Computing for {session_tag} ")
    explanations_ci = judge.evaluate_consistency(context, rollback, exp_gen)

    #VE ,LLM judge
    print(f"\n  [VE] Computing for {session_tag} ")
    judge_result = judge.evaluate(
        trace=trace,
        explanation=explanation_text,
        target_agent=context["target_agent"],
        target_timestamp=context["target_timestamp"]
    )

    #all metrics
    metrics = FidelityMetrics.compute_all(
        explanations_ci=explanations_ci,
        explanation=explanation_text,
        causes=causes,
        ve_score=judge_result["ve_score"]
    )
    return metrics


#SCRIBE pipeline
def run_scribe(trace) -> tuple:

    target_agent = "Acheteur"
    steps = trace["steps"]
    buyer_steps = [s for s in steps if s["agent"] == target_agent]
    target_timestamp = buyer_steps[0]["timestamp"] if buyer_steps else 1

    rollback     = Rollback(trace)
    context      = rollback.extract_context(target_agent, target_timestamp)
    alt_gen      = AlternativeGenerator()
    alternatives = alt_gen.generate(context, rollback)
    cf_sim       = CounterfactualSimulator()
    cf_traces    = cf_sim.simulate_all(context, alternatives, rollback)
    cause_id     = CauseIdentifier()
    causes       = cause_id.identify(context, cf_traces)
    exp_gen      = ExplanationGenerator()
    explanation  = exp_gen.generate(context=context, causes=causes,
                                    cf_traces=cf_traces,
                                    session_id=trace["session_id"])

    metrics = evaluate(trace, context, causes,
                       explanation["text_explanation"], "SCRIBE")
    return metrics, context, causes


#baseline textual

def run_baseline_textual(trace) -> dict:

    target_agent = "Acheteur"
    steps = trace["steps"]
    buyer_steps = [s for s in steps if s["agent"] == target_agent]
    target_timestamp = buyer_steps[0]["timestamp"] if buyer_steps else 1

    rollback = Rollback(trace)
    context  = rollback.extract_context(target_agent, target_timestamp)

    baseline = TextualBaseline()
    explanation, causes = baseline.run(context, trace, trace["session_id"])

    metrics = evaluate(trace, context, causes,
                       explanation["text_explanation"], "Baseline-Textual")
    return metrics


#baseline temporal t=1

def run_baseline_temporal(trace) -> dict:
    print("\n" + "="*60)
    print("  STEP 4 : Baseline Temporal (t=1 rollback)")
    print("="*60)

    baseline = TemporalBaseline()
    explanation, causes, cf_traces, context = baseline.run(
        trace=trace,
        original_target_agent="Acheteur",
        session_id=trace["session_id"]
    )

    metrics = evaluate(trace, context, causes,
                       explanation["text_explanation"], "Baseline-Temporal-t1")
    return metrics


#print Table 8
def print_table8(scribe_m, textual_m, temporal_m):
    header = f"table 8 — baseline Comparison"
    row_fmt = "  {:<30} {:>8} {:>8} {:>8} {:>12}"

    print(header)
    print(row_fmt.format("System", "CI", "VE", "CO", "Global"))
    print(row_fmt.format(
        "SCRIBE (full pipeline)",
        scribe_m["CI"], scribe_m["VE"], scribe_m["CO"], scribe_m["global_score"]
    ))
    print(row_fmt.format(
        "Baseline Textual",
        textual_m["CI"], textual_m["VE"], textual_m["CO"], textual_m["global_score"]
    ))
    print(row_fmt.format(
        "Baseline Temporal (t=1)",
        temporal_m["CI"], temporal_m["VE"], temporal_m["CO"], temporal_m["global_score"]
    ))

    #save as JSON
    os.makedirs("outputs", exist_ok=True)
    table = {
        "SCRIBE":              scribe_m,
        "Baseline_Textual":    textual_m,
        "Baseline_Temporal_t1": temporal_m,
    }
    with open("outputs/table8_comparison.json", "w") as f:
        json.dump(table, f, indent=2)

    #save as plain text
    lines = [
        "table 8 — baseline comparison",
        f"{'System':<30} {'CI':>8} {'VE':>8} {'CO':>8} {'Global':>12}",
        "-"*66,
        f"{'SCRIBE (full pipeline)':<30} {scribe_m['CI']:>8} {scribe_m['VE']:>8} {scribe_m['CO']:>8} {scribe_m['global_score']:>12}",
        f"{'Baseline Textual':<30} {textual_m['CI']:>8} {textual_m['VE']:>8} {textual_m['CO']:>8} {textual_m['global_score']:>12}",
        f"{'Baseline Temporal (t=1)':<30} {temporal_m['CI']:>8} {temporal_m['VE']:>8} {temporal_m['CO']:>8} {temporal_m['global_score']:>12}",
    ]
    with open("outputs/table8_comparison.txt", "w") as f:
        f.write("\n".join(lines))



if __name__ == "__main__":
    trace_path, trace = run_mas()
    scribe_m, context, causes = run_scribe(trace)
    textual_m  = run_baseline_textual(trace)
    temporal_m = run_baseline_temporal(trace)
    print_table8(scribe_m, textual_m, temporal_m)
