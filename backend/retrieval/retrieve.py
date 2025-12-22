from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def select_dataset(question: str):
    q = question.lower()

    if "tax" in q:
        return "data/taxation.txt"
    if "license" in q or "shop" in q:
        return "data/licensing.txt"
    if "inspect" in q or "enter property" in q:
        return "data/inspection.txt"
    if any(w in q for w in ["pee", "spit", "urinate"]):
        return "data/public_nuisance.txt"

    return "data/waste.txt"


class Retriever:
    def __init__(self, clauses):
        self.clauses = clauses
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(clauses)

    def retrieve(self, query, top_k=3):
        scores = self.vectorizer.transform([query])
        similarities = cosine_similarity(scores, self.matrix)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]
        docs = [self.clauses[i] for i in top_indices]

        best_score = float(similarities[top_indices[0]]) if top_indices.size > 0 else 0.0
        return docs, best_score
