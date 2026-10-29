
# enregistre toutes les interactions entre N agents dans une trace JSON
# trace est la ground truth utilisee par CEMA pour generer les explications
import json
import os
import sys
from datetime import datetime
from config import OUTPUT_TRACES


def _safe_print(text: str):
    """Affiche du texte en gérant les caractères non supportés par la console Windows."""
    encoded = text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
    print(encoded)


class TraceLogger:
   
    def __init__(self, scenario: str, agents: list):
        self.scenario   = scenario
        self.agents     = [a.name for a in agents]
        self.steps      = []
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(OUTPUT_TRACES, exist_ok=True)

    def log_step(self, step: dict):
        self.steps.append(step)
        _safe_print(f"\n[t={step['timestamp']}] {step['agent']} "
                    f"(recu de : {step.get('sender','?')})")
        _safe_print(f"  THOUGHT     : {step['thought'][:90]}...")
        _safe_print(f"  ACTION      : {step['action'][:90]}...")
        _safe_print(f"  OBSERVATION : {step['observation'][:90]}...")

    def save(self, final_outcome: str = "") -> str:
     
        trace = {
            "session_id"     : self.session_id,
            "scenario"       : self.scenario,
            "agents"         : self.agents,
            "nb_agents"      : len(self.agents),
            "steps"          : self.steps,
            "total_steps"    : len(self.steps),
            "final_outcome"  : final_outcome,
            "timestamp_saved": datetime.now().isoformat(),
        }
        path = os.path.join(OUTPUT_TRACES, f"{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        print(f"\n  Trace sauvegardee : {path}")
        return path

    @staticmethod
    def load(path: str) -> dict:
        #charger une trace depuis un fichier JSON
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)