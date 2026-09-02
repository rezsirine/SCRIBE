
#Approche : Paradigme ReAct (Reason + Act)
# agent LLM autonome suivant le paradigme ReAct(Thought -> Action -> Observation)
# Supporte N agents dans le meme systeme


import re
from groq import Groq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE, MAX_TOKENS


class Agent:
    #     Agent LLM autonome suivant le paradigme ReAct
    # Attributes:
    #     name         : identifiant de l'agent (ex: "Vendeur", "Acheteur", "Mediateur")
    #     role         : description du role
    #     objective    : objectif dans le scenario
    #     memory       : historique des echanges
    #     react_trace  : trace structuree Thought/Action/Observation
 

    def __init__(self, name: str, role: str, objective: str):
        self.name        = name
        self.role        = role
        self.objective   = objective
        self.memory      = []
        self.react_trace = []

        # Client Groq
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.model_name  = GROQ_MODEL
        print(f"  Agent '{name}' initialise - modele : {self.model_name}")

        # System prompt ReAct
        self.system_prompt = f"""
Tu es {self.name}, un agent autonome dans un systeme multi-agents
Ton role      : {self.role}
Ton objectif  : {self.objective}

Tu participes a une negociation avec PLUSIEURS agents
A chaque tour, structure OBLIGATOIREMENT ta reponse ainsi :

[THOUGHT] ton raisonnement interne (analyse la situation, pense a ta strategie)
[ACTION]  ton message envoye aux autres agents (ce que tu dis)
[OBSERVATION] ce que tu retiens et observes de la situation actuelle

Exemple :
[THOUGHT] Le vendeur demande 200 euros. C'est trop cher. Je dois negocier.
[ACTION] Je propose 130 euros maximum pour ce produit.
[OBSERVATION] Le vendeur va probablement refuser et contre-proposer.

Important : reste dans ton role et travaille vers ton objectif.
"""

    def _call_llm(self, messages: list) -> str:
     #Appelle Groq avec la liste de messages
        groq_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                groq_messages.append({"role": "system",    "content": msg.content})
            elif isinstance(msg, HumanMessage):
                groq_messages.append({"role": "user",      "content": msg.content})
            elif isinstance(msg, AIMessage):
                groq_messages.append({"role": "assistant", "content": msg.content})

        response = self.groq_client.chat.completions.create(
            model       = self.model_name,
            messages    = groq_messages,
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
        )
        return response.choices[0].message.content

    def step(self, incoming_message: str, timestamp: int,
             sender: str = "Systeme") -> dict:
        
        # Effectue un pas de raisonnement ReAct

        # Args:
        #     incoming_message  message recu
        #     timestamp        pas de temps courant
        #     sender            nom de l'agent qui envoie le message

        # Returns:
        #     dict avec thought, action, observation
        
        # Message contextualise avec le nom de l'expediteur
        contextualized = f"[Message de {sender}] : {incoming_message}"
        self.memory.append(HumanMessage(content=contextualized))

        # Construire le prompt
        messages = [SystemMessage(content=self.system_prompt)] + self.memory

        # Appel LLM
        raw = self._call_llm(messages)

        # Parser les balises ReAct
        thought     = self._extract_tag(raw, "THOUGHT")
        action      = self._extract_tag(raw, "ACTION")
        observation = self._extract_tag(raw, "OBSERVATION")

        # Sauvegarder en memoire
        self.memory.append(AIMessage(content=raw))

        # Enregistrer dans la trace ReAct
        step_record = {
            "timestamp"  : timestamp,
            "agent"      : self.name,
            "sender"     : sender,
            "incoming"   : incoming_message,
            "thought"    : thought,
            "action"     : action,
            "observation": observation,
            "raw"        : raw,
        }
        self.react_trace.append(step_record)
        return step_record

    def _extract_tag(self, text: str, tag: str) -> str:
    #    Extrait le contenu entre [TAG] et le prochain tag
        pattern = rf'\[{tag}\](.*?)(?=\[(?:THOUGHT|ACTION|OBSERVATION)\]|$)'
        match   = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def get_context_at(self, timestamp: int) -> dict:
        # retourne l'etat de l'agent au moment t0 pour le rollback
        return {
            "agent"    : self.name,
            "role"     : self.role,
            "objective": self.objective,
            "history"  : [s for s in self.react_trace if s["timestamp"] < timestamp],
            "timestamp": timestamp,
        }

    def reset(self):
        # Remet l'agent a zero pour les simulations contrefactuelles
        self.memory      = []
        self.react_trace = []