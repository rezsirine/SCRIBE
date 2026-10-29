"""
BASELINE 2 — Temporal (t=1) Baseline
Runs the FULL SCRIBE pipeline but forces the rollback
to t=1 regardless of the actual target timestamp.
"""

import os
import json
from explanation.rollback import Rollback
from explanation.alternatives import AlternativeGenerator
from explanation.counterfactual import CounterfactualSimulator
from explanation.causeIdentifier import CauseIdentifier
from explanation.explanation_gen import ExplanationGenerator
from config import OUTPUT_EXPLANATIONS


class TemporalBaseline:

    def __init__(self):
        os.makedirs(OUTPUT_EXPLANATIONS, exist_ok=True)

    def run(self, trace: dict, original_target_agent: str, session_id: str) -> dict:

        #find first step for the target agent  t=1 
        steps = trace.get("steps", [])

        #find the first step at timestamp=1 /any agent
        first_steps = [s for s in steps if s["timestamp"] == 1]
        if not first_steps:
            #use the second step if t=1 doesn't exist
            sorted_steps = sorted(steps, key=lambda s: s["timestamp"])
            first_steps = [sorted_steps[1]] if len(sorted_steps) > 1 else [sorted_steps[0]]

        #use the target agent's step at t=1 if available
        target_step_t1 = next(
            (s for s in first_steps if s["agent"] == original_target_agent),
            first_steps[0]
        )

        forced_target_agent = target_step_t1["agent"]
        forced_timestamp = target_step_t1["timestamp"]

        print(f"  [baseline temporal t=1] Using agent={forced_target_agent}, t={forced_timestamp}")

        #run SCRIBE pipeline with  t=1 
        rollback = Rollback(trace)
        context = rollback.extract_context(forced_target_agent, forced_timestamp)

        alt_gen = AlternativeGenerator()
        alternatives = alt_gen.generate(context, rollback)

        cf_sim = CounterfactualSimulator()
        cf_traces = cf_sim.simulate_all(context, alternatives, rollback)

        cause_id = CauseIdentifier()
        causes = cause_id.identify(context, cf_traces)

        exp_gen = ExplanationGenerator()
        explanation = exp_gen.generate(
            context=context,
            causes=causes,
            cf_traces=cf_traces,
            session_id=session_id + "_temporal_t1"
        )

        explanation["baseline"] = "temporal_t1"
        explanation["forced_timestamp"] = forced_timestamp
        explanation["original_target_agent"] = original_target_agent

        path = os.path.join(OUTPUT_EXPLANATIONS, f"baseline_temporal_{session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(explanation, f, ensure_ascii=False, indent=2)

        print(f"  [baseline temporal t=1] Saved to {path}")
        return explanation, causes, cf_traces, context
