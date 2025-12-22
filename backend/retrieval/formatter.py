def format_answer(text: str, question: str) -> str:
    """
    Final formatting layer.
    Returns clean, human-readable output without labels.
    """
    if not text:
        return "No relevant municipal regulation was found."

    # Remove any accidental markdown labels
    text = text.replace("**Answer:**", "")
    text = text.replace("**Penalty:**", "")
    text = text.replace("Answer:", "")
    text = text.replace("Penalty:", "")

    return text.strip()
