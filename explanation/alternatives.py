#génère des actions alternatives pour l'agent cible
import json
import re
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE, K_ALTERNATIVES
from groq_retry import groq_call_with_retry

class AlternativeGenerator:

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = GROQ_MODEL

    def generate(self, context: dict, rollback_obj) -> list:
        history_text = rollback_obj.format_history_as_text(context["history_all"])
        prompt = f"""
Tu es {context['target_agent']} dans cette negociation.
Historique jusqu'a t={context['t0']} :
{history_text}

Ta decision reelle a t={context['target_timestamp']} : "{context['real_action']}"

Quelles autres actions aurais-tu pu prendre ?
Genere exactement {K_ALTERNATIVES} alternatives realistes.
Reponds UNIQUEMENT au format JSON :
{{
  "alternatives": [
    {{"id": 1, "action": "...", "rationale": "..."}},
    {{"id": 2, "action": "...", "rationale": "..."}},
    {{"id": 3, "action": "...", "rationale": "..."}}
  ]
}}
"""
        response = groq_call_with_retry(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=[{"role": "system", "content": "Tu es un assistant qui genere du JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
        )
        raw = response.choices[0].message.content
        raw = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(raw)
            return data.get("alternatives", [])
        except Exception as e:
            print(f"  [Erreur JSON Alternatives] : {e}")
            return [{"id": 1, "action": "Proposer un compromis", "rationale": "Option par defaut"}]
