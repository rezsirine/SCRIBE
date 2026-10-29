
#dual-dimension causal analysis
# deux dimensions (comme CEMA) 
#teleologique  
#mecanique     


import re
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

class CauseIdentifier:
    def __init__(self):
        #client Groq pour noter l'action reelle
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = GROQ_MODEL

    def _score_real_action(self, context: dict) -> int:
        """
        Note l'action REELLE avec le meme prompt/rubric que celui utilise
        pour noter les alternatives contrefactuelles (OBJECTIFS_ATTEINTS).
        Corrige l'asymetrie : avant, l'action reelle avait un score fixe de 5
        alors que les alternatives etaient notees dynamiquement par le LLM.
        """
        prompt = f"""
Tu es {context.get('target_agent', 'un agent')} dans cette negociation.
Ton action a t={context.get('target_timestamp', 't')} : "{context.get('real_action', '')}"

Sur une echelle de 0 a 10 (10 = objectif totalement atteint),
quel score donnerais-tu a cette action par rapport a ton objectif ?

Reponds UNIQUEMENT avec :
[OBJECTIFS_ATTEINTS] score de 0 a 10
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Tu es un evaluateur objectif de decisions de negociation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,   # basse temperature : on veut un score stable, pas creatif
            max_tokens=50,
        )
        raw = response.choices[0].message.content
        numbers = re.findall(r'\b(10|[0-9])\b', raw)
        return int(numbers[0]) if numbers else 5  

    #compare la trace reelle avec les CF(ai) pour identifier les causes et les influences entre les N agents


    def identify(self, context: dict, cf_traces: list) -> dict:
      
        #identifie les causes et influences
        real_action  = context["real_action"]
        other_agents = context["other_agents"]

        teleological = self._analyze_teleological(real_action, cf_traces, context)
        mechanical   = self._analyze_mechanical(real_action, cf_traces, other_agents)

        causes = {
            "real_action"  : real_action,
            "teleological" : teleological,
            "mechanical"   : mechanical,
            "best_cf"      : self._best_cf(cf_traces),
            "worst_cf"     : self._worst_cf(cf_traces),
            "impact_score" : self._impact_score(cf_traces),
            "agents_influenced": self._agents_influenced(cf_traces, other_agents),
        }

        print(f"\n  [Causes] Teleologique : {teleological['summary'][:80]}")
        print(f"  [Causes] Mecanique    : {mechanical['summary'][:80]}")
        print(f"  [Causes] Impact score : {causes['impact_score']:.3f}")
        print(f"  [Causes] Agents influences : {causes['agents_influenced']}")

        return causes

    def _analyze_teleological(self, real_action: str, cf_traces: list, context: dict = None) -> dict:

        #dimension teleologique : compare les objectifs atteints
        # entre la trace reelle et les traces contrefactuelles  
        real_score = self._score_real_action(context) if context else 5

        cf_scores  = [cf["objective_score"] for cf in cf_traces]
        avg_cf     = sum(cf_scores) / len(cf_scores) if cf_scores else 5

        better = [cf for cf in cf_traces if cf["objective_score"] > real_score]
        worse  = [cf for cf in cf_traces if cf["objective_score"] < real_score]

        if avg_cf > real_score + 1:
            summary = (f"La decision '{real_action[:60]}' etait sous-optimale — "
                       f"{len(better)} alternatives auraient mieux servi l'objectif.")
        elif avg_cf < real_score - 1:
            summary = (f"La decision '{real_action[:60]}' etait la meilleure — "
                       f"les alternatives auraient moins bien servi l'objectif.")
        else:
            summary = (f"La decision '{real_action[:60]}' etait equivalente "
                       f"aux alternatives en termes d'objectifs.")

        return {
            "summary"        : summary,
            "real_score"     : real_score,
            "avg_cf_score"   : round(avg_cf, 2),
            "better_count"   : len(better),
            "worse_count"    : len(worse),
            "cf_scores"      : cf_scores,
        }

    def _analyze_mechanical(self, real_action: str,
                            cf_traces: list, other_agents: list) -> dict:

        #dimension mecanique:identifie comment la decision a influence
        # le comportement de chaque autre agent
        influence_per_agent = {}

        for agent in other_agents:
            reactions = []
            for cf in cf_traces:
                reaction = cf.get("reactions", {}).get(agent, "")
                if reaction and len(reaction) > 15:
                    reactions.append({
                        "alternative": cf["alternative_action"],
                        "reaction"   : reaction[:200],
                        "cf_score"   : cf["objective_score"],
                    })
            influence_per_agent[agent] = reactions

        influenced = [
            ag for ag, reactions in influence_per_agent.items()
            if len(reactions) > 0
        ]

        if influenced:
            summary = (f"La decision a influence differemment : "
                       f"{', '.join(influenced)}.")
        else:
            summary = "Aucune influence inter-agents significative detectee."

        return {
            "summary"             : summary,
            "influenced_agents"   : influenced,
            "influence_per_agent" : influence_per_agent,
        }

    def _agents_influenced(self, cf_traces: list, other_agents: list) -> list:
        """
        Retourne la liste des agents reellement influences par la decision.

        CORRECTIF : avant, un agent etait marque "influence" des que sa reaction
        faisait plus de 15 caracteres — un seuil trop permissif qui capturait
        presque toute reponse non-vide. Desormais, un agent est considere
        influence seulement si ses reactions VARIENT de facon significative
        d'une alternative a l'autre (coherent avec la logique de variance
        deja utilisee pour impact_score).
        """
        influenced = []

        for agent in other_agents:
            reactions = [
                cf.get("reactions", {}).get(agent, "")
                for cf in cf_traces
                if cf.get("reactions", {}).get(agent, "")
            ]
            if len(reactions) < 2:
                continue  

#similarite de Jaccard entre chaque paire de reactions
            def tokenize(t):
                return set(re.findall(r'\b\w+\b', t.lower()))

            tokens = [tokenize(r) for r in reactions]
            sims = []
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):
                    inter = len(tokens[i] & tokens[j])
                    union = len(tokens[i] | tokens[j])
                    sims.append(inter / union if union > 0 else 0)

            avg_similarity = sum(sims) / len(sims) if sims else 1.0

#si les reactions sont tres similaires d'une alternative a l'autre
            if avg_similarity < 0.5:
                influenced.append(agent)

        return influenced

    def _best_cf(self, cf_traces: list) -> dict:
        if not cf_traces:
            return {}
        return max(cf_traces, key=lambda cf: cf["objective_score"])

    def _worst_cf(self, cf_traces: list) -> dict:
        if not cf_traces:
            return {}
        return min(cf_traces, key=lambda cf: cf["objective_score"])

    def _impact_score(self, cf_traces: list) -> float:

#variance des scores objectifs 
        if len(cf_traces) < 2:
            return 0.0
        scores   = [cf["objective_score"] for cf in cf_traces]
        mean     = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return round(variance, 3)