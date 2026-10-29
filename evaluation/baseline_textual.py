"""
textual Baseline
Uses only ExplanationGenerator._generate_text() without:
  - counterfactual simulation
  - causal graph
  - alternative generation
Equivalent to a naive LLM-summariser applied to the raw trace.
"""

import os
import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, OUTPUT_EXPLANATIONS


class TextualBaseline:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = GROQ_MODEL
        os.makedirs(OUTPUT_EXPLANATIONS, exist_ok=True)

    def _build_minimal_causes(self, context: dict) -> dict:
        return {
            "best_cf": {},
            "teleological": {
                "summary": "No counterfactual analysis performed (textual baseline)."
            },
            "mechanical": {
                "summary": "No mechanical analysis performed (textual baseline)."
            },
            "agents_influenced": [],
            "impact_score": 0.0,
        }
# generate a causal explanation using only the raw trace history without counterfactual alternatives
        
    def generate_text_only(self, context: dict) -> str:
        
        history_lines = []
        for step in context.get("history_all", []):
            history_lines.append(
                f"[t={step['timestamp']}] {step['agent']}: {step.get('action', '')[:200]}"
            )
        history_text = "\n".join(history_lines) if history_lines else "No history available."

        all_agents = ", ".join(context.get("all_agents", []))
        target = context.get("target_agent", "Unknown")
        real_action = context.get("real_action", "Unknown action")
        t = context.get("target_timestamp", "?")

        prompt = f"""You are an XAI system. Your task is to explain an agent's decision
based ONLY on the conversation history — no counterfactual analysis is available.

Agents present    : {all_agents}
Agent to explain  : {target}
Real decision     : "{real_action}"
Timestamp         : t={t}

Conversation history (up to t-1):
{history_text}

Generate a causal explanation in 4-6 sentences answering:
Why did {target} take this decision at t={t}?

Start directly with the explanation. Do not mention that you lack counterfactual data.
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an XAI system. Generate precise causal explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
#generate text explanation (no counterfactual) and return explanation dict compatible with SCRIBE evaluator
    def run(self, context: dict, trace: dict, session_id: str) -> dict:

        text = self.generate_text_only(context)
        causes = self._build_minimal_causes(context)

        explanation = {
            "session_id": session_id,
            "baseline": "textual",
            "target_agent": context["target_agent"],
            "target_timestamp": context["target_timestamp"],
            "all_agents": context["all_agents"],
            "real_action": context["real_action"],
            "text_explanation": text,
            "graph_path": None,   # No graph in textual baseline
            "impact_score": 0.0,
            "teleological": causes["teleological"],
            "mechanical": causes["mechanical"],
            "agents_influenced": [],
        }

        path = os.path.join(OUTPUT_EXPLANATIONS, f"baseline_textual_{session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(explanation, f, ensure_ascii=False, indent=2)

        print(f"  [baseline textual] Saved to {path}")
        return explanation, causes
