import math
from collections import Counter
from retrieval.preprocess import tokenize

class TFIDF:
    def __init__(self, documents):
        self.documents = documents
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.vocab = self.build_vocab()
        self.idf = self.compute_idf()

    def build_vocab(self):
        vocab = set()
        for tokens in self.doc_tokens:
            vocab.update(tokens)
        return sorted(list(vocab))

    def tf(self, tokens):
        tf_dict = Counter(tokens)
        total = len(tokens)
        return {term: tf_dict[term] / total for term in tf_dict}

    def compute_idf(self):
        idf = {}
        total_docs = len(self.doc_tokens)

        for term in self.vocab:
            doc_count = sum(1 for tokens in self.doc_tokens if term in tokens)
            idf[term] = math.log((total_docs + 1) / (doc_count + 1)) + 1

        return idf
    def vectorize(self, tokens):
        tf_vals = self.tf(tokens)
        vector = []

        for term in self.vocab:
            vector.append(tf_vals.get(term, 0) * self.idf.get(term, 0))

        return vector
def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)
