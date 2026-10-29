#generation de l'explication multi-format
#texte en langage naturel
#graphe d'influences entre N agents
#score d'impact contrefactuel

import os
import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, OUTPUT_EXPLANATIONS
from groq_retry import groq_call_with_retry

try:
    import networkx as nx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class ExplanationGenerator:

    def __init__(self):
        self.client     = Groq(api_key=GROQ_API_KEY)
        self.model_name = GROQ_MODEL
        os.makedirs(OUTPUT_EXPLANATIONS, exist_ok=True)

    def generate(self, context: dict, causes: dict,
                 cf_traces: list, session_id: str) -> dict:
#genere l'explication complete multi-format
        print("  Generation de l'explication finale")

        text  = self._generate_text(context, causes, cf_traces)
        graph = self._generate_graph(context, causes, session_id)
        score = causes["impact_score"]

        explanation = {
            "session_id"        : session_id,
            "target_agent"      : context["target_agent"],
            "target_timestamp"  : context["target_timestamp"],
            "all_agents"        : context["all_agents"],
            "real_action"       : context["real_action"],
            "question"          : (f"Pourquoi {context['target_agent']} "
                                   f"a pris cette decision a t="
                                   f"{context['target_timestamp']} ?"),
            "text_explanation"  : text,
            "graph_path"        : graph,
            "impact_score"      : score,
            "teleological"      : causes["teleological"],
            "mechanical"        : causes["mechanical"],
            "agents_influenced" : causes["agents_influenced"],
        }
        path = os.path.join(OUTPUT_EXPLANATIONS,
                            f"explanation_{session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(explanation, f, ensure_ascii=False, indent=2)

        print(f"\n  TEXTE :\n  {text}")
        print(f"\n  SCORE D'IMPACT    : {score}")
        print(f"\n  AGENTS INFLUENCES : {causes['agents_influenced']}")
        print(f"\n  GRAPHE            : {graph}")

        return explanation

    def _generate_text(self, context: dict, causes: dict,
                       cf_traces: list) -> str:
#genere l'explication en langage naturel 
        best_cf      = causes.get("best_cf", {})
        tele         = causes["teleological"]
        mech         = causes["mechanical"]
        all_agents   = ", ".join(context["all_agents"])
        influenced   = ", ".join(causes["agents_influenced"]) or "aucun"

        prompt = f"""
Tu es un systeme d'explication XAI inspire de CEMA
Genere une explication causale claire en 4-6 phrases

Agents presents    : {all_agents}
Agent explique     : {context['target_agent']}
Decision reelle    : "{context['real_action']}"
Agents influences  : {influenced}

Teleologique : {tele['summary']}
Mecanique    : {mech['summary']}
Meilleure alternative : "{best_cf.get('alternative_action', 'N/A')}"
Score objectif meilleure alt. : {best_cf.get('objective_score', '?')}/10

Pourquoi {context['target_agent']} a-t-il pris cette decision
a t={context['target_timestamp']} dans cette negociation multi-agents ?

Commence directement par l'explication causale.
Mentionne l'influence sur les autres agents.
"""
        response = groq_call_with_retry(
            self.client.chat.completions.create,
            model      = self.model_name,
            messages   = [
                {"role": "system",
                 "content": "Tu es un systeme XAI. Genere des explications causales precises."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.3,
            max_tokens  = 500,
        )
        return response.choices[0].message.content.strip()

    def _generate_graph(self, context: dict, causes: dict,
                        session_id: str) -> str:
        if not HAS_NETWORKX:
            return "NetworkX non installe"

        os.makedirs("outputs/graphs", exist_ok=True)

        G           = nx.DiGraph()
        target      = context["target_agent"]
        all_agents  = context["all_agents"]
        influenced  = causes["agents_influenced"]
        score       = causes["impact_score"]

#ajouter les noeuds
        for agent in all_agents:
            G.add_node(agent)

#ajouter les arcs d'influence
        mech = causes["mechanical"]
        for agent in all_agents:
            if agent != target:
#arc target:agent si influence detectee
                if agent in influenced:
                    weight = max(1.0, min(4.0, 1.0 + score * 2))
                    G.add_edge(target, agent,
                               weight=weight,
                               influenced=True)
                else:
                   #arc leger pour montrer l'interaction meme sans influence
                    G.add_edge(target, agent,
                               weight=0.5,
                               influenced=False)

        plt.figure(figsize=(9, 7))
        seed = abs(hash(session_id)) % 9999
        pos  = nx.spring_layout(G, seed=seed, k=2.0)

        node_colors = [
            "#e74c3c" if n == target else "#3498db"
            for n in G.nodes()
        ]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=2500, alpha=0.92)
        nx.draw_networkx_labels(G, pos, font_size=11,
                                font_weight="bold", font_color="white")

        edges_strong = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get("influenced", False)]
        edges_weak   = [(u, v) for u, v, d in G.edges(data=True)
                        if not d.get("influenced", False)]

        if edges_strong:
            weights_strong = [G[u][v]["weight"] for u, v in edges_strong]
            nx.draw_networkx_edges(G, pos, edgelist=edges_strong,
                                   arrows=True, arrowsize=25,
                                   width=weights_strong,
                                   edge_color="#e74c3c", alpha=0.8)
        if edges_weak:
            nx.draw_networkx_edges(G, pos, edgelist=edges_weak,
                                   arrows=True, arrowsize=15,
                                   width=0.8,
                                   edge_color="#95a5a6",
                                   style="dashed", alpha=0.6)

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend = [
            Patch(color="#e74c3c", label=f"Agent explique : {target}"),
            Patch(color="#3498db", label="Autres agents"),
            Line2D([0], [0], color="#e74c3c", linewidth=2,
                   label="Influence forte"),
            Line2D([0], [0], color="#95a5a6", linewidth=1,
                   linestyle="--", label="Interaction simple"),
        ]
        plt.legend(handles=legend, loc="upper left", fontsize=9)

        plt.title(
            f"Graphe d'influences CEMA — {target}\n"
            f"t={context['target_timestamp']} | "
            f"Impact={score:.3f} | "
            f"Agents influences : {', '.join(influenced) or 'aucun'}",
            fontsize=12, fontweight="bold"
        )
        plt.axis("off")
        plt.tight_layout()

        path = f"outputs/graphs/graph_{session_id}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path