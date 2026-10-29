#   CI , VE ,CO,ST 

import re


class FidelityMetrics:

    @staticmethod
    def compute_ci(explanations: list) -> float:
      
        # coherence interne : similarite entre N explications
        # score entre 0  et 1 

        if len(explanations) < 2:
            return 1.0

        def tokenize(t):
            return set(re.findall(r'\b\w+\b', t.lower()))

        tokens = [tokenize(e) for e in explanations]
        sims   = []
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                inter = len(tokens[i] & tokens[j])
                union = len(tokens[i] | tokens[j])
                sims.append(inter / union if union > 0 else 0)

        return round(sum(sims) / len(sims), 3) if sims else 0.0

    @staticmethod
    def compute_co(explanation: str, causes: dict) -> float:
        key_elements = []
        tele = causes.get("teleological", {})
        mech = causes.get("mechanical", {})

        if tele.get("summary"):
            key_elements += tele["summary"].split()[:6]
        if mech.get("summary"):
            key_elements += mech["summary"].split()[:6]
        for agent in causes.get("agents_influenced", []):
            key_elements.append(agent)

        if not key_elements:
            return 0.5

        exp_lower = explanation.lower()
        covered   = sum(
            1 for w in key_elements
            if len(w) > 3 and w.lower() in exp_lower
        )
        return round(covered / len(key_elements), 3)

    @staticmethod
    def compute_all(explanations_ci: list, explanation: str,
                    causes: dict, ve_score: float) -> dict:
        ci     = FidelityMetrics.compute_ci(explanations_ci)
        co     = FidelityMetrics.compute_co(explanation, causes)
        global_score = round((ci + ve_score + co) / 3, 3)

        report = {
            "CI"           : ci,
            "VE"           : round(ve_score, 3),
            "CO"           : co,
            "global_score" : global_score,
            "interpretation": FidelityMetrics._interpret(global_score),
        }

        print(f"    CI  : {ci}")
        print(f"    VE   : {round(ve_score, 3)}")
        print(f"    CO   : {co}")
        print(f"    Score global           : {global_score}")
        print(f"    Interpretation         : {report['interpretation']}")

        return report

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 0.8:
            return "Explication tres fidele — haute confiance"
        elif score >= 0.6:
            return "Explication moderement fidele — confiance acceptable"
        elif score >= 0.4:
            return "Explication partiellement fidele — a verifier"
        else:
            return "Explication peu fidele — confiance faible"