# utils/enhancer.py

def enhance_answer(text: str) -> str:
    """
    Convert flat answer into structured bullet points
    """
    if not text or len(text.split()) < 15:
        return text

    sentences = [s.strip() for s in text.split(".") if s.strip()]

    if len(sentences) >= 2:
        return (
            f"• {sentences[0]}.\n"
            f"• {sentences[1]}."
        )

    return text


def add_legal_softener(text: str) -> str:
    """
    Prevent over-claiming; adds legal caution
    """
    lowered = text.lower()
    if "may" in lowered or "depending" in lowered:
        return text

    return "Depending on local municipal bylaws, " + text[0].lower() + text[1:]


def question_prefix(question: str) -> str:
    """
    Add context-aware opening phrase
    """
    q = question.lower()

    if "penalt" in q or "fine" in q:
        return "Under municipal regulations, "
    if "what is" in q or "define" in q:
        return "Municipal laws generally provide that "
    if "action" in q or "what happens" in q:
        return "As per municipal authority powers, "

    return ""
def is_in_domain(question: str) -> bool:
    q = question.lower()

    keywords = [
        "municipal", "waste", "garbage", "trash", "penalty", "fine",
        "spit", "urinate", "dump", "segregation", "license",
        "tax", "inspection", "public", "nuisance"
    ]

    return any(k in q for k in keywords)
from difflib import SequenceMatcher

def is_repetitive(new: str, old: str, threshold=0.75) -> bool:
    if not old:
        return False
    return SequenceMatcher(None, new, old).ratio() > threshold
