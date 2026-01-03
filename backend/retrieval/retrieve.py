# retrieval/retrieve.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from retrieval.intent import detect_intent

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



# =============================
# POST-RETRIEVAL FILTER
# =============================
def filter_irrelevant(docs, intent):
    keywords = {
        "building_property": ["boundary", "encroachment", "construction", "permit", "land"],
        "waste": ["waste", "dumping", "segregation"],
        "public_nuisance": ["spitting", "urination", "noise", "nuisance"],
    }

    if intent not in keywords:
        return docs  # 🚑 DO NOT FILTER

    filtered = [
        d for d in docs
        if any(k in d.lower() for k in keywords[intent])
    ]

    return filtered if filtered else docs  # 🚑 fallback


   

# 🔑 DATASET ROUTER (UPDATED)
def select_dataset(question: str) -> str:
    q = question.lower()
    INTENT_DATASET_MAP = {
    "waste": "data/waste.txt",
    "public_nuisance": "data/public_nuisance.txt",
    "building_property": "data/building_property.txt",
    "licensing": "data/licensing.txt",
    "taxation": "data/taxation.txt",
    "inspection": "data/inspection.txt",
    "civic_services": "data/civic_services.txt",
    "notices": "data/notices_enforcement.txt",
}
    # 🧱 Building & Property
    if any(w in q for w in [
        "boundary", "encroachment", "construction", "building",
        "unauthorized", "demolition", "occupancy", "setback", "property"
    ]):
        return "data/building_property.txt"

    # 🚮 Waste Management
    if any(w in q for w in [
        "waste", "garbage", "dump", "segregation",
        "litter", "burning", "bulk waste"
    ]):
        return "data/waste.txt"

    # 🚫 Public Nuisance
    if any(w in q for w in [
        "spit", "urinate", "urination", "noise",
        "nuisance", "mosquito", "stagnant", "defecation"
    ]):
        return "data/public_nuisance.txt"

    # 🏪 Licensing & Trade
    if any(w in q for w in [
        "license", "shop", "trade", "vendor",
        "hawker", "hotel", "restaurant"
    ]):
        return "data/licensing.txt"

    # 💰 Taxation
    if any(w in q for w in [
        "tax", "property tax", "arrears",
        "assessment", "penalty", "recovery"
    ]):
        return "data/taxation.txt"

    # 👮 Inspection & Enforcement
    if any(w in q for w in [
        "inspection", "inspect", "officer",
        "seizure", "entry", "raid"
    ]):
        return "data/inspection.txt"

    # 🚰 Civic Services
    if any(w in q for w in [
        "water", "sewer", "drain", "road",
        "street light", "park", "public toilet"
    ]):
        return "data/civic_services.txt"

    # 📜 Notices & Legal Action
    if any(w in q for w in [
        "notice", "show cause", "eviction",
        "penalty notice", "hearing", "appeal"
    ]):
        return "data/notices_enforcement.txt"

    # 🧠 DEFAULT (SAFE FALLBACK)
    return "data/waste.txt"
