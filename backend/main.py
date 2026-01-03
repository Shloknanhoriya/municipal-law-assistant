from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from utils.intent import normalize_question
from utils.enhancer import is_in_domain, is_repetitive
from retrieval.intent import detect_intent
from retrieval.retrieve import Retriever, select_dataset, filter_irrelevant
from retrieval.refine import refine_answer
from retrieval.formatter import format_answer
from summarizer.textrank import summarize_text
from model.rewriter import rewrite_text

# -----------------------------
# GLOBAL STATE
# -----------------------------
SESSION_MEMORY = []
MAX_MEMORY = 3
LAST_ANSWER = ""

# -----------------------------
# APP INIT (SINGLE INSTANCE)
# -----------------------------
app = FastAPI(title="Municipal Law Assistant Backend")

# -----------------------------
# CORS (MUST BE BEFORE ROUTES)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://municipal-law-assistant.vercel.app",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# GLOBAL OPTIONS HANDLER (CRITICAL)
# -----------------------------
@app.options("/{path:path}")
async def options_handler(path: str, request: Request):
    return {}

# -----------------------------
# SCHEMA
# -----------------------------
class Query(BaseModel):
    question: str

# -----------------------------
# MEMORY INJECTION
# -----------------------------
def inject_memory(question: str, intent: str) -> str:
    relevant = [m for m in SESSION_MEMORY if m["intent"] == intent]
    if not relevant:
        return question
    return relevant[-1]["question"] + " " + question

# -----------------------------
# API ROUTE (WITH /api PREFIX)
# -----------------------------
@app.post("/api/ask")
def ask(query: Query):
    global LAST_ANSWER

    try:
        normalized_question = normalize_question(query.question)
        intent = detect_intent(normalized_question)
        normalized_question = inject_memory(normalized_question, intent)

        if not is_in_domain(normalized_question):
            return {
                "answer": (
                    "This assistant is designed to answer questions related to "
                    "municipal laws and civic regulations."
                ),
                "show_context": False,
                "context": []
            }

        if intent == "public_nuisance":
            factual = (
                "Under municipal regulations, public nuisance activities such as "
                "public urination or spitting in public places are prohibited and "
                "may attract fines imposed by the local authority."
            )
            return {
                "answer": rewrite_text(factual),
                "show_context": False,
                "context": []
            }

        dataset_path = select_dataset(normalized_question)
        clauses = open(dataset_path, encoding="utf-8").read().split("\n\n")
        retriever = Retriever(clauses)

        docs, score = retriever.retrieve(normalized_question, top_k=5)

        if score < 0.15:
            return {
                "answer": (
                    "Municipal regulations on this matter may vary by jurisdiction. "
                    "Please consult the local municipal authority."
                ),
                "show_context": False,
                "context": []
            }

        docs = filter_irrelevant(docs, intent)

        if not docs:
            return {
                "answer": "No relevant municipal regulation was found.",
                "show_context": False,
                "context": []
            }

        combined = " ".join(docs)
        summarized = summarize_text(combined, max_sentences=2)
        factual = refine_answer([summarized], normalized_question)
        rewritten = rewrite_text(factual) if factual else factual
        final_answer = format_answer(rewritten, normalized_question)

        if is_repetitive(final_answer, LAST_ANSWER):
            final_answer = (
                "Municipal regulations require compliance, and violations "
                "may attract penalties imposed by local authorities."
            )

        LAST_ANSWER = final_answer

        SESSION_MEMORY.append({
            "question": query.question,
            "intent": intent
        })
        if len(SESSION_MEMORY) > MAX_MEMORY:
            SESSION_MEMORY.pop(0)

        return {
            "answer": final_answer,
            "show_context": True,
            "context": docs
        }

    except Exception as e:
        print("ERROR:", e)
        return {
            "answer": "Internal processing error.",
            "show_context": False,
            "context": []
        }

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
