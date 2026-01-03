def detect_intent(question: str) -> str:
    q = question.lower()

    if any(w in q for w in [
        "boundary", "encroach", "encroachment",
        "neighbor", "neighbour", "property line",
        "my land", "my plot"
    ]):
        return "building_property"

    if any(w in q for w in [
        "waste", "dump", "garbage", "litter"
    ]):
        return "waste"

    if any(w in q for w in [
        "spit", "urinate", "noise", "mosquito"
    ]):
        return "public_nuisance"

    return "unknown"
