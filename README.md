# 🤖 RAG-Based Document Q&A Chatbot using LLMs

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF/DOCX 
files and ask questions from them using powerful LLMs via Groq.

---

## 📌 Project Overview

This project delivers two full implementations of a RAG pipeline: loading and 
chunking documents, embedding them into a vector store, retrieving relevant 
context, and generating precise answers using LLMs. The objective is to enable 
intelligent document Q&A without hallucination by grounding responses strictly 
in uploaded content.

---

## 🧰 Tech Stack

- **Language:** Python
- **Libraries:** LangChain, ChromaDB, HuggingFace, Streamlit, FastAPI
- **LLM Provider:** Groq (Llama 4, Kimi, Qwen, GPT-OSS)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Environment:** Local / Any Python 3.9+ environment

---

## 📁 Project Structure

```
RAG_Chatbot/
│
├── Standalone_Streamlit/
│   └── app.py               # Single-file RAG app using Streamlit
│
├── Streamlit_FastAPI/
│   ├── backend/
│   │   └── main.py          # FastAPI backend (upload + ask endpoints)
│   └── frontend/
│       └── app.py           # Streamlit frontend consuming the API
│
├── requirements.txt         # All dependencies
├── .gitignore
└── README.md
```

---

## 🔄 Workflow Summary

### 1. Document Ingestion
Upload PDF or DOCX files which are parsed page by page and chunked using 
`RecursiveCharacterTextSplitter` (chunk size: 700, overlap: 100).

### 2. Embedding & Vector Store
Chunks are embedded using HuggingFace `all-MiniLM-L6-v2` and stored in 
**ChromaDB** for fast similarity search.

### 3. Retrieval
On each query, the top 3 most relevant chunks are retrieved from ChromaDB 
using semantic similarity.

### 4. Generation
Retrieved context is passed to a Groq-hosted LLM via LangChain's RAG chain. 
The model answers **only from the provided context** — if the answer isn't 
found, it says so.

---

## 🚀 Two Implementations

### ✅ Standalone Streamlit
Single-file app — everything in one place. Best for quick demos.

```bash
streamlit run Standalone_Streamlit/app.py
```

### ✅ Streamlit + FastAPI
Decoupled architecture — FastAPI handles the backend logic, Streamlit is 
just the UI. Best for production-style projects.

```bash
# Terminal 1 — Start backend
uvicorn Streamlit_FastAPI.backend.main:app --reload

# Terminal 2 — Start frontend
streamlit run Streamlit_FastAPI/frontend/app.py
```

---

## ⚙️ Setup

1. Clone the repo
```bash
git clone https://github.com/dhurai-aiml/RAG_Chatbot.git
cd RAG_Chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root folder
```
GROQ_API_KEY=your_groq_api_key_here
```
---

## 📈 Key Features

- Upload multiple PDFs or DOCX files at once
- Incremental indexing — new files are added to existing vector store
- Choose from multiple LLMs (Llama 4, Kimi K2, Qwen, GPT-OSS)
- Answers grounded strictly in document context — no hallucination
- Persistent chat history within session

---

## 🧑‍💻 Author

**Dhurai Murugan S** — Fresher | AIML & Gen AI Enthusiast

**Skills:** Python • Machine Learning • Deep Learning • NLP • Generative AI • RAG Systems • Agentic AI

[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/dhurai-aiml)
