import math
import re
from collections import defaultdict
from retrieval.preprocess import tokenize, clean_text

def split_sentences(text: str):
    # Basic sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def sentence_similarity(s1_tokens, s2_tokens):
    if not s1_tokens or not s2_tokens:
        return 0.0
    common = set(s1_tokens).intersection(set(s2_tokens))
    if not common:
        return 0.0
    return len(common) / (math.log(len(s1_tokens)+1) + math.log(len(s2_tokens)+1))

def textrank(sentences, max_iter=20, d=0.85):
    # Build tokens
    tokenized = [tokenize(s) for s in sentences]
    n = len(sentences)

    # Build similarity graph
    graph = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                graph[i][j] = sentence_similarity(tokenized[i], tokenized[j])

    # Initialize scores
    scores = [1.0] * n

    # PageRank iterations
    for _ in range(max_iter):
        new_scores = [1-d] * n
        for i in range(n):
            for j in range(n):
                if graph[j][i] > 0:
                    denom = sum(graph[j])
                    if denom != 0:
                        new_scores[i] += d * (graph[j][i] / denom) * scores[j]
        scores = new_scores

    return scores

def summarize_text(text, max_sentences=2):
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    scores = textrank(sentences)
    ranked = sorted(
        zip(scores, sentences),
        reverse=True,
        key=lambda x: x[0]
    )

    top = [s for _, s in ranked[:max_sentences]]
    return " ".join(top)
