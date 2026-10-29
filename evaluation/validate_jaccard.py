import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explanation.rollback import Rollback
from explanation.alternatives import AlternativeGenerator
from explanation.counterfactual import CounterfactualSimulator


def tokenize(text: str) -> set:
    return set(re.findall(r'\b\w+\b', text.lower()))


def mean_pairwise_jaccard(reactions: list) -> float:
    tokens = [tokenize(r) for r in reactions]
    sims = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            inter = len(tokens[i] & tokens[j])
            union = len(tokens[i] | tokens[j])
            sims.append(inter / union if union > 0 else 0.0)
    return round(sum(sims) / len(sims), 3) if sims else None


def validate():
    trace_dir = "outputs/traces"
    trace_files = sorted([
        f for f in os.listdir(trace_dir) if f.endswith(".json")
    ])

    
    all_similarities = []
    results = []

    for filename in trace_files:
        path = os.path.join(trace_dir, filename)
        try:
            with open(path, encoding="utf-8") as f:
                trace = json.load(f)
        except Exception as e:
            print(f"  SKIP {filename}: {e}")
            continue

        steps = trace.get("steps", [])
        if not steps:
            continue

        session_id = trace.get("session_id", filename)
        print(f"\n  Session: {session_id}")

        # target
        target_agent     = steps[0]["agent"]
        target_timestamp = steps[0]["timestamp"]

        try:
            rollback     = Rollback(trace)
            context      = rollback.extract_context(target_agent, target_timestamp)
            alternatives = AlternativeGenerator().generate(context, rollback)
            cf_traces    = CounterfactualSimulator().simulate_all(
                context, alternatives, rollback
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        other_agents = context.get("other_agents", [])
        session_pairs = []

        for agent in other_agents:
            reactions = [
                cf.get("reactions", {}).get(agent, "")
                for cf in cf_traces
                if cf.get("reactions", {}).get(agent, "")
            ]
            if len(reactions) < 2:
                continue

            sim = mean_pairwise_jaccard(reactions)
            if sim is None:
                continue

            influenced = sim < 0.5
            all_similarities.append(sim)
            session_pairs.append({
                "agent": agent,
                "jaccard": sim,
                "influenced": influenced
            })
            print(f"    {agent:<20} Jaccard={sim:.3f}  "
                  f"{'INFLUENCED' if influenced else 'NOT influenced'}")

        results.append({
            "session": session_id,
            "target": target_agent,
            "pairs": session_pairs
        })

    if not all_similarities:
        print("no data computed")
        return

    min_sim  = min(all_similarities)
    max_sim  = max(all_similarities)
    mean_sim = round(sum(all_similarities) / len(all_similarities), 3)

    print(f"  Total agent-decision pairs : {len(all_similarities)}")
    print(f"  Similarity range           : [{min_sim:.3f}, {max_sim:.3f}]")
    print(f"  global mean                : {mean_sim:.3f}")
    print()

    for threshold in [0.25, 0.28, 0.30, 0.40, 0.50]:
        flagged = sum(1 for s in all_similarities if s < threshold)
        pct = flagged / len(all_similarities) * 100
        print(f"  Threshold {threshold:.2f} → "
              f"{flagged}/{len(all_similarities)} pairs flagged ({pct:.0f}%)")

    # save
    os.makedirs("outputs", exist_ok=True)
    output = {
        "total_pairs": len(all_similarities),
        "min": min_sim,
        "max": max_sim,
        "mean": mean_sim,
        "all_values": all_similarities,
        "sessions": results,
        "threshold_analysis": {
            str(t): sum(1 for s in all_similarities if s < t)
            for t in [0.25, 0.28, 0.30, 0.40, 0.50]
        }
    }
    with open("outputs/jaccard_validation.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    lines = [
        "Jaccard Threshold Validation",
        f"Total pairs : {len(all_similarities)}",
        f"Range       : [{min_sim:.3f}, {max_sim:.3f}]",
        f"Mean        : {mean_sim:.3f}",
        "",
        "Threshold analysis:",
    ]
    for t in [0.25, 0.28, 0.30, 0.40, 0.50]:
        n = sum(1 for s in all_similarities if s < t)
        lines.append(f"  {t:.2f} → {n}/{len(all_similarities)} pairs flagged")

    with open("outputs/jaccard_validation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n  saved: outputs/jaccard_validation.json")
    print("  saved: outputs/jaccard_validation.txt")
    return output


if __name__ == "__main__":
    validate()
