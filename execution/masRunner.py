
# Approche :Orchestration Round-Robin
# Gere les tours de dialogue entre N agents (pas juste 2)

# Scenario S1 : Negociation a 3 agents
#   - Vendeur   : veut vendre au prix le plus haut
#   - Acheteur  : veut acheter au prix le plus bas
#   - Mediateur : aide a trouver un accord equitable

# Structure des tours (round-robin entre N agents) :
#   t=0 : Acheteur repond au Vendeur
#   t=1 : Mediateur repond a l'Acheteur
#   t=2 : Vendeur repond au Mediateur
#   t=3 : Acheteur repond au Vendeur


from execution.agent import Agent
from execution.traceLogger import TraceLogger
from config import MAX_TURNS


class MASRunner:
    # Orchestrateur multi-agents generique pour N agents
    # Les agents parlent en round-robin : A -> B -> C -> A -> B -> C ...

    def __init__(self, agents: list, scenario: str = "S1_negociation_3agents"):
        # Args:
        #     agents   : liste d'objets Agent (dans l'ordre de parole)
        #     scenario : nom du scenario

        self.agents   = agents
        self.logger   = TraceLogger(scenario=scenario, agents=agents)
        self.trace    = None
        self.nb_agents = len(agents)

    def run(self, initial_message: str = None) -> str:
      
        # Lance la session multi-agents.

        # Le premier agent (agents[0]) envoie le message initial
        # au deuxieme agent (agents[1]).
        # Ensuite, les agents parlent en round-robin.

        # Returns:
        #     chemin du fichier de trace sauvegarde
    
        print(f"  LANCEMENT SESSION : {self.logger.session_id}")
        print(f"  Scenario          : {self.logger.scenario}")
        print(f"  Agents ({self.nb_agents})        : {[a.name for a in self.agents]}")

        if initial_message is None:
            initial_message = (
                "Bonjour a tous ! Je propose ce produit electronique "
                "a 200 euros. J'attends vos offres et remarques."
            )

        outcome = "En cours"

        # Le message initial vient de agents[0]
        current_message = initial_message
        sender_idx      = 0   # index de celui qui vient de parler
        receiver_idx    = 1   # index de celui qui doit repondre

        for t in range(MAX_TURNS):
            sender   = self.agents[sender_idx]
            receiver = self.agents[receiver_idx]

            # Le receveur repond
            step = receiver.step(
                incoming_message = current_message,
                timestamp        = t,
                sender           = sender.name
            )
            self.logger.log_step(step)

            # Detection de fin de negociation
            action_lower = step["action"].lower()
            mots_accord  = ["accord", "accepte", "deal", "d'accord",
                            "marche", "convenu", "parfait", "vendu"]
            mots_refus   = ["refuse", "impossible", "inacceptable",
                            "arret", "stop", "termine", "rupture"]

            if any(kw in action_lower for kw in mots_accord):
                outcome = f"Accord trouve au pas t={t} entre {sender.name} et {receiver.name}"
                print(f"\n  ACCORD : {outcome}")
                break
            elif any(kw in action_lower for kw in mots_refus):
                outcome = f"Echec au pas t={t} — {receiver.name} a refuse"
                print(f"\n  ECHEC  : {outcome}")
                break

            # Passage au prochain receveur (round-robin)
            current_message = step["action"]
            sender_idx      = receiver_idx
            receiver_idx    = (receiver_idx + 1) % self.nb_agents

        else:
            outcome = f"Session terminee apres {MAX_TURNS} tours sans conclusion"
            print(f"\n  FIN : {outcome}")

        # Sauvegarder la trace
        trace_path  = self.logger.save(final_outcome=outcome)
        self.trace  = TraceLogger.load(trace_path)
        return trace_path