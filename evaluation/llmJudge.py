# Approche : LLM-as-a-Judge (4eme agent non participant)
# Evalue la fidelite de l'explication par rapport a la trace reelle
# Inspire de Papageorgiou et al. (2025)


import re
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, N_CONSISTENCY


class LLMJudge:

    def __init__(self):
        self.client     = Groq(api_key=GROQ_API_KEY)
        self.model_name = GROQ_MODEL

    def evaluate(self, trace: dict, explanation: str,
                 target_agent: str, target_timestamp: int) -> dict:
      
        # Evalue la fidelite de l'explication par rapport a la trace reelle
        # Le juge n'a pas participe a la simulation

        # Extraire les pas pertinents (±2 autour de la decision)
        relevant = [
            s for s in trace["steps"]
            if abs(s["timestamp"] - target_timestamp) <= 2
        ]
        excerpt = "\n".join([
            f"[t={s['timestamp']}] {s['agent']} -> Action : {s['action'][:150]}"
            for s in relevant
        ])

        all_agents = ", ".join(trace.get("agents", []))

        prompt = f"""
Tu es un evaluateur XAI independant.
Tu n'as PAS participe a la negociation.
Les agents presents etaient : {all_agents}

 EXTRAIT TRACE REELLE (autour de t={target_timestamp}) 
{excerpt}

 EXPLICATION A EVALUER 
{explanation}

Evalue si l'explication est fidele a ce qui s'est reellement passe.

Reponds EXACTEMENT ainsi :
COHERENCE: oui/partiellement/non
CONFIRME: elements de la trace qui confirment l'explication
CONTREDIT: elements manquants ou contradictoires
SCORE: nombre de 0 a 10
JUSTIFICATION: une phrase de synthese
"""
        response = self.client.chat.completions.create(
            model      = self.model_name,
            messages   = [
                {"role": "system",
                 "content": "Tu es un juge XAI impartial. Evalue objectivement."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.1,
            max_tokens  = 400,
        )
        raw = response.choices[0].message.content

        score_raw = self._field(raw, "SCORE")
        try:
            score_int = int(re.search(r'\d+', score_raw).group())
            ve_score  = min(1.0, max(0.0, score_int / 10.0))
        except:
            ve_score = 0.5

        result = {
            "ve_score"     : ve_score,
            "raw_score"    : score_raw,
            "coherence"    : self._field(raw, "COHERENCE"),
            "confirmed_by" : self._field(raw, "CONFIRME"),
            "contradicted" : self._field(raw, "CONTREDIT"),
            "justification": self._field(raw, "JUSTIFICATION"),
        }

        print(f"\n  [Juge LLM] Score VE     : {ve_score} ({score_raw}/10)")
        print(f"  [Juge LLM] Coherence    : {result['coherence']}")
        print(f"  [Juge LLM] Justification: {result['justification'][:100]}")

        return result

    def evaluate_consistency(self, context: dict,
                             rollback_obj,
                             exp_gen) -> list:

        # Genere N_CONSISTENCY explications independantes
        # pour mesurer la coherence interne

        print(f"\n  [CI] {N_CONSISTENCY} repetitions...")
        explanations = []

        from explanation.alternatives import AlternativeGenerator
        from explanation.counterfactual import CounterfactualSimulator
        from explanation.causeIdentifier import CauseIdentifier

        for i in range(N_CONSISTENCY):
            alts   = AlternativeGenerator().generate(context, rollback_obj)
            cfs    = CounterfactualSimulator().simulate_all(context, alts, rollback_obj)
            causes = CauseIdentifier().identify(context, cfs)
            text   = exp_gen._generate_text(context, causes, cfs)
            explanations.append(text)
            print(f"    Repetition {i+1}/{N_CONSISTENCY} OK")

        return explanations

    def _field(self, text: str, field: str) -> str:
        pattern = rf'{field}:\s*(.+?)(?=\n[A-Z]+:|$)'
        match   = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""