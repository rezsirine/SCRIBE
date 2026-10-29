
# Approche :orchestration Round-Robin
# gere les tours de dialogue entre N agents 

# Scenario1:negociation a 3 agents
# vendeur qui veut vendre au prix le plus haut
#acheteur:veut acheter au prix le plus bas
#mediateur quiaide a trouver un accord equitable

# Structure des tours (round-robin entre N agents) :
#   t=0 : Acheteur repond au Vendeur
#   t=1 : Mediateur repond a l'Acheteur
#   t=2 : Vendeur repond au Mediateur
#   t=3 : Acheteur repond au Vendeur


from execution.agent import Agent
from execution.traceLogger import TraceLogger
from config import MAX_TURNS


class MASRunner:
    #les agents parlent en round-robin : A -> B -> C -> A -> B -> C 

    def __init__(self, agents: list, scenario: str = "S1_negociation_3agents"):
        
        self.agents   = agents
        self.logger   = TraceLogger(scenario=scenario, agents=agents)
        self.trace    = None
        self.nb_agents = len(agents)

    def run(self, initial_message: str = None) -> str:
      
        
        print(f"  LANCEMENT SESSION : {self.logger.session_id}")
        print(f"  Scenario          : {self.logger.scenario}")
        print(f"  Agents ({self.nb_agents})        : {[a.name for a in self.agents]}")

        if initial_message is None:
            initial_message = (
                "Bonjour a tous ! Je propose ce produit electronique "
                "a 200 euros. J'attends vos offres et remarques."
            )

        outcome = "En cours"

       
        current_message = initial_message
        sender_idx      = 0  
        receiver_idx    = 1  

        for t in range(MAX_TURNS):
            sender   = self.agents[sender_idx]
            receiver = self.agents[receiver_idx]

           
            step = receiver.step(
                incoming_message = current_message,
                timestamp        = t,
                sender           = sender.name
            )
            self.logger.log_step(step)

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

            # passage au prochain receveur (round-robin)
            current_message = step["action"]
            sender_idx      = receiver_idx
            receiver_idx    = (receiver_idx + 1) % self.nb_agents

        else:
            outcome = f"Session terminee apres {MAX_TURNS} tours sans conclusion"
            print(f"\n  FIN : {outcome}")

        # sauvegarder la trace
        trace_path  = self.logger.save(final_outcome=outcome)
        self.trace  = TraceLogger.load(trace_path)
        return trace_path