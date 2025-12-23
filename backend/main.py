from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from utils.intent import normalize_question
from utils.enhancer import is_in_domain, is_repetitive
from retrieval.intent import detect_intent
from retrieval.retrieve import Retriever, select_dataset
from retrieval.refine import refine_answer
from retrieval.formatter import format_answer
from summarizer.textrank import summarize_text
from model.rewriter import rewrite_text

# -----------------------------
# GLOBAL STATE
# -----------------------------
LAST_ANSWER = ""

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="Municipal Law Assistant Backend")

# -----------------------------
# CORS (✅ FIXED)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://municipal-law-assistant.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# SCHEMA
# -----------------------------
class Query(BaseModel):
    question: str

# -----------------------------
# API
# -----------------------------
@app.post("/ask")
def ask(query: Query):
    global LAST_ANSWER

    try:
        # 1️⃣ Normalize informal language
        normalized_question = normalize_question(query.question)

        # 2️⃣ Domain guard
        if not is_in_domain(normalized_question):
            return {
                "answer": (
                    "This assistant is designed to answer questions related to "
                    "municipal laws and civic regulations. I may not be able to "
                    "provide information on this topic."
                ),
                "source": "Domain Guard",
                "show_context": False,
                "context": []
            }

        # 3️⃣ Detect intent
        intent = detect_intent(normalized_question)

        # 4️⃣ Public nuisance shortcut
        if intent == "public_nuisance":
            factual_answer = (
                "Under municipal regulations, public nuisance activities such as "
                "public urination or spitting in public places are prohibited and "
                "may attract fines imposed by the local authority."
            )

            rewritten = rewrite_text(factual_answer)

            return {
                "answer": rewritten,
                "source": "Rule-based Public Nuisance Handling",
                "show_context": False,
                "context": []
            }

        # 5️⃣ Dataset routing
        dataset_path = select_dataset(normalized_question)

        # 6️⃣ Load clauses dynamically
        clauses = open(dataset_path).read().split("\n\n")
        retriever = Retriever(clauses)

        # 7️⃣ Retrieve clauses
        docs, score = retriever.retrieve(normalized_question, top_k=3)

        if not docs:
            return {
                "answer": "No relevant municipal regulation was found for this query.",
                "source": "Legal Retrieval System",
                "show_context": False,
                "context": []
            }

        # 8️⃣ Summarization
        combined = " ".join(docs)
        summarized = summarize_text(combined, max_sentences=2)

        # 9️⃣ Rule-based refinement
        factual_answer = refine_answer([summarized], normalized_question)

        # 🔟 Neural rewriting
        rewritten = rewrite_text(factual_answer)

        # 1️⃣1️⃣ Final formatting
        final_answer = format_answer(rewritten, normalized_question)

        # 1️⃣2️⃣ Anti-repetition guard
        if is_repetitive(final_answer, LAST_ANSWER):
            final_answer = (
                "Under municipal regulations, compliance with municipal rules "
                "is mandatory, and violations may lead to penalties as prescribed "
                "by the local authority."
            )

        LAST_ANSWER = final_answer

        return {
            "answer": final_answer,
            "source": "Legal Retrieval + Neural Rewriter",
            "show_context": True,
            "context": docs
        }

    except Exception as e:
        print("ERROR:", e)
        return {
            "answer": "Internal processing error.",
            "source": "System",
            "show_context": False,
            "context": []
        }
@app.get("/health")
def health():
    return {"status": "ok"}
