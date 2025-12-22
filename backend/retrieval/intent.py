def detect_intent(question: str) -> str:
    q = question.lower()

    if any(w in q for w in ["pee", "urinate", "urination", "spit", "spitting"]):
        return "public_nuisance"

    if any(w in q for w in ["penalty", "fine", "punishment", "what happens"]):
        return "penalty"

    if any(w in q for w in ["mandatory", "required", "allowed", "can i"]):
        return "rules"

    return "general"
