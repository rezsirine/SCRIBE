
#3. Approche : LLM-Based Self-Simulation
# Difference cle avec CEMA original :
#   CEMA utilise IGP2 (modele probabiliste externe)
#   Notre version : le LLM lui-meme simule les traces alternatives
#   pour TOUS les autres agents presents dans la scene


import re
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


class CounterfactualSimulator:
    # Pour chaque alternative, simule la reaction de TOUS les autres agents
    # Innovation : auto-simulation LLM sans modele probabiliste externe

    def __init__(self):
        self.client     = Groq(api_key=GROQ_API_KEY)
        self.model_name = GROQ_MODEL

    def simulate(self, context: dict, alternative: dict,
                 rollback_obj) -> dict:
        
        # Simule la trace contrefactuelle pour une alternative donnee
        # Inclut les reactions de TOUS les autres agents (pas juste un)

        # Args:
        #     context      : contexte C(t0)
        #     alternative  : {"id", "action", "rationale"}
        #     rollback_obj : instance Rollback

        # Returns:
        #     dict CF(ai) avec les reactions simulees de chaque agent
       
        history_text  = rollback_obj.format_history_as_text(context["history_all"])
        other_agents  = context["other_agents"]
        others_str    = ", ".join(other_agents)

        # Construire les balises de reaction pour chaque agent
        reaction_tags = "\n".join([
            f"[REACTION_{ag}] comment {ag} aurait reagi"
            for ag in other_agents
        ])

        prompt = f"""
Tu es {context['target_agent']} dans cette negociation.
Les autres agents presents sont : {others_str}

 HISTORIQUE JUSQU'A t={context['t0']} 
{history_text}

 SCENARIO CONTREFACTUEL 
Au lieu de dire : "{context['real_action']}"
Tu aurais dit   : "{alternative['action']}"

 SIMULATION DEMANDEE 
Simule en detail comment CHAQUE agent aurait reagi a cette action alternative,
et comment la conversation aurait continue sur 2-3 tours.

Structure OBLIGATOIRE de ta reponse :
{reaction_tags}
[SUITE_CONVERSATION] les 2-3 tours suivants entre tous les agents
[RESULTAT_FINAL] le resultat final de la negociation
[OBJECTIFS_ATTEINTS] score de 0 a 10 (10 = objectif totalement atteint)
"""
        response = self.client.chat.completions.create(
            model      = self.model_name,
            messages   = [
                {"role": "system",
                 "content": "Tu es un simulateur multi-agents. Sois precis sur les reactions de chaque agent."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.4,
            max_tokens  = 1200,
        )
        raw = response.choices[0].message.content

        # Parser les reactions de chaque agent
        reactions = {}
        for agent in other_agents:
            tag      = f"REACTION_{agent}"
            reaction = self._extract(raw, tag)
            reactions[agent] = reaction

        cf_trace = {
            "alternative_id"    : alternative["id"],
            "alternative_action": alternative["action"],
            "reactions"         : reactions,            # dict {agent: reaction}
            "conversation_flow" : self._extract(raw, "SUITE_CONVERSATION"),
            "final_result"      : self._extract(raw, "RESULTAT_FINAL"),
            "objective_score"   : self._extract_score(raw, "OBJECTIFS_ATTEINTS"),
            "raw_simulation"    : raw,
        }

        print(f"\n  Simulation CF [{alternative['id']}] : "
              f"{alternative['action'][:60]}...")
        for ag, reaction in reactions.items():
            print(f"    Reaction {ag} : {reaction[:60]}...")
        print(f"    Resultat : {cf_trace['final_result'][:60]}...")
        print(f"    Score    : {cf_trace['objective_score']}/10")

        return cf_trace

    def simulate_all(self, context: dict, alternatives: list,
                     rollback_obj) -> list:
        # Simule toutes les alternatives
        return [
            self.simulate(context, alt, rollback_obj)
            for alt in alternatives
        ]

    def _extract(self, text: str, tag: str) -> str:
        # Extrait le contenu d'une balise
        # Pattern flexible pour les tags avec underscore
        escaped = re.escape(tag)
        pattern = rf'\[{escaped}\](.*?)(?=\[[A-Z_]+\]|$)'
        match   = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_score(self, text: str, tag: str) -> int:
        """Extrait le score numerique."""
        section = self._extract(text, tag)
        numbers = re.findall(r'\b(10|[0-9])\b', section)
        return int(numbers[0]) if numbers else 5