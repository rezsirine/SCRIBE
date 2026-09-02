# GROQ : Backend LLM

import os
from dotenv import load_dotenv

load_dotenv()

#  GROQ 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_YOUR_KEY_HERE")
GROQ_MODEL   = "llama-3.3-70b-versatile"

#  PARAMÈTRES LLM 
TEMPERATURE  = 0.3
MAX_TOKENS   = 1000

#  PARAMÈTRES CEMA 
K_ALTERNATIVES = 3    # nombre d'alternatives contrefactuelles par agent
N_CONSISTENCY  = 3    # répétitions pour mesurer la cohérence interne
MAX_TURNS      = 20   # tours max de dialogue entre agents
#MAX_TURNS      = 20   

#  CHEMINS DE SORTIE 
OUTPUT_TRACES       = "outputs/traces/"
OUTPUT_EXPLANATIONS = "outputs/explanations/"
OUTPUT_GRAPHS       = "outputs/graphs/"