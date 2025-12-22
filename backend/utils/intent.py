def normalize_question(question: str) -> str:
    q = question.lower()

    if "pee" in q or "urinate" in q:
        return "What are the penalties for public urination in public places?"

    if "spit" in q:
        return "What are the penalties for spitting in public places?"

    return question
