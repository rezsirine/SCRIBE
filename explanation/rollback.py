

# 1. Approche : State Backtracking
class Rollback:
   
    # Extrait le contexte C(t0) pour l'analyse contrefactuelle.
    
    def __init__(self, trace: dict):
        self.trace = trace
        self.steps = trace["steps"]

    def extract_context(self, target_agent: str, target_timestamp: int) -> dict:
        t0 = target_timestamp - 1 if target_timestamp > 0 else 0
        history_all = [s for s in self.steps if s["timestamp"] <= t0]
        history_target = [s for s in self.steps if s["agent"] == target_agent and s["timestamp"] <= t0]
        target_step = next((s for s in self.steps if s["agent"] == target_agent and s["timestamp"] == target_timestamp), None)

        all_agents = list(set(s["agent"] for s in self.steps))
        other_agents = [a for a in all_agents if a != target_agent]

        context = {
            "target_agent": target_agent,
            "target_timestamp": target_timestamp,
            "t0": t0,
            "real_action": target_step["action"] if target_step else "Action inconnue",
            "real_thought": target_step["thought"] if target_step else "",
            "history_all": history_all,
            "history_target": history_target,
            "scenario": self.trace.get("scenario", ""),
            "session_id": self.trace.get("session_id", ""),
            "all_agents": all_agents,
            "other_agents": other_agents
        }
        return context

    def format_history_as_text(self, history: list) -> str:
        lines = []
        for step in history:
            lines.append(f"[t={step['timestamp']}] {step['agent']} :\n  Action: {step.get('action', '')[:150]}")
        return "\n\n".join(lines)
