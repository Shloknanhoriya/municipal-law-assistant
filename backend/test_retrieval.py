from retrieval.retrieve import Retriever

clauses = open("data/clauses.txt").read().split("\n\n")

retriever = Retriever(clauses)

results = retriever.retrieve("What is penalty for illegal dumping?")

for r in results:
    print("-", r)
