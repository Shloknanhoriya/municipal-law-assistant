# Municipal Law Assistant  
### RAG-based Question Answering System (Academic NLP Project)

---

## 📌 Overview

The **Municipal Law Assistant** is a domain-specific, Retrieval-Augmented Generation (RAG) system designed to answer questions related to **municipal laws and civic regulations**.

Unlike generic chatbots, this system is:
- 🔒 Domain-restricted (municipal laws only)
- 🧠 Built using **classical NLP + custom-trained neural models**
- 📚 Explainable, deterministic, and academically aligned

This project was developed as part of an **NLP academic project** and focuses on **information retrieval, summarization, and controlled generation**, not open-ended conversation.

---

## 🎯 Key Features

- ✔️ Question normalization for informal user input
- ✔️ Domain guard to reject out-of-scope questions
- ✔️ Intent detection for legal categorization
- ✔️ Dataset routing based on query type
- ✔️ TF-IDF based legal clause retrieval
- ✔️ Extractive summarization (TextRank)
- ✔️ Rule-based legal refinement
- ✔️ Custom-trained Seq2Seq neural rewriter (with attention)
- ✔️ Answer de-duplication & formatting
- ✔️ Optional legal context transparency

---

## 🧠 System Architecture (Flowchart)

User Question
↓
Question Normalization
↓
Domain Guard
↓
Intent Detection
↓
Dataset Routing
↓
TF-IDF Legal Retrieval
↓
Extractive Summarization
↓
Rule-based Legal Refinement
↓
Neural Rewriter (Seq2Seq + Attention)
↓
Answer De-duplication & Formatting
↓
Final Answer + Optional Legal Context

yaml
Copy code

---

## 🏗️ Technology Stack

### Frontend
- React + TypeScript
- Vite
- Tailwind CSS
- shadcn/ui

### Backend
- FastAPI
- Scikit-learn (TF-IDF)
- Custom TextRank summarizer
- PyTorch (Seq2Seq + Attention model)
- Rule-based NLP pipelines

---

## 🚫 What This Project Is NOT

- ❌ Not a ChatGPT clone
- ❌ Not dependent on cloud LLM APIs
- ❌ Not a prompt-based chatbot
- ❌ Not a black-box model

This is a **fully engineered NLP system** built from first principles.

---

## 📦 Project Structure (Simplified)

frontend/
│── src/
│ ├── components/
│ ├── pages/
│ ├── App.tsx
│ └── main.tsx
│
backend/
│── data/ # Legal datasets
│── model/ # Trained neural rewriter
│── retrieval/ # TF-IDF + intent logic
│── summarizer/ # TextRank
│── utils/ # Guards & normalization
│── main.py # FastAPI entry point

yaml
Copy code

---

## 🚀 Running the Project Locally

### Frontend
```bash
cd frontend
npm install
npm run dev
Backend
bash
Copy code
cd backend
uvicorn main:app --reload --port 8000
🎓 Academic Value
This project demonstrates:

Practical NLP pipeline design

Classical IR + neural hybrid systems

Controlled generation for legal domains

Explainable AI principles

Real-world deployability

👤 Author
Shlok Nanhoriya
B.Tech CSE | NLP & AI
Academic Project – Municipal Law Assistant