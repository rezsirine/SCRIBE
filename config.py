import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "KEY")
GROQ_MODEL       = "openai/gpt-oss-120b"   
GROQ_JUDGE_MODEL = "qwen/qwen3.8-27b"     

# hyperparamètres de génération LLM 
TEMPERATURE  = 0.3    
MAX_TOKENS   = 1000   

# paramètres du moteur contrefactuel SCRIBE 
K_ALTERNATIVES = 3   
N_CONSISTENCY  = 3    
MAX_TURNS      = 20   

#répertoires de sauvegarde des résultats 
OUTPUT_TRACES       = "outputs/traces/"
OUTPUT_EXPLANATIONS = "outputs/explanations/"
OUTPUT_GRAPHS       = "outputs/graphs/"
