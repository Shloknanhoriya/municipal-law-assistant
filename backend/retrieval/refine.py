def refine_answer(contexts, question):
    question = question.lower()
    base = contexts[0] if contexts else ""

    if any(w in question for w in ["spit", "spitting", "urinate", "pee"]):
        return (
            base +
            " Such acts are treated as public nuisance under municipal regulations "
            "and may attract fines or penalties."
        )

    if "illegal dumping" in question or "dumping" in question:
        return (
            base +
            " Municipal authorities may impose fines or initiate legal action "
            "for such violations."
        )

    if "segregation" in question:
        return (
            base +
            " This requirement applies to both households and commercial establishments."
        )

    if "penalty" in question or "fine" in question:
        return (
            base +
            " The exact penalty may vary depending on local municipal bylaws."
        )

    return base
