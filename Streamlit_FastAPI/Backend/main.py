from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import io, os, re, tempfile
import pypdf
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import Docx2txtLoader

load_dotenv()
app = FastAPI()
VECTOR_DB = None
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
UPLOADED_FILES: set = set()


class Question(BaseModel):
    question: str
    model_name: str


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    global VECTOR_DB
    docs = []
    skipped = []

    for file in files:
        if file.filename in UPLOADED_FILES:
            skipped.append(file.filename)
            continue

        suffix = file.filename.split(".")[-1].lower()

        if suffix == "pdf":
            pdf = pypdf.PdfReader(io.BytesIO(await file.read()))
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file.filename, "page": i}
                    ))

        elif suffix == "docx":
            content = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                loader = Docx2txtLoader(tmp_path)
                for doc in loader.load():
                    doc.metadata["source"] = file.filename
                    docs.append(doc)
            finally:
                os.unlink(tmp_path)

        UPLOADED_FILES.add(file.filename)

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    if chunks:
        if VECTOR_DB is None:
            VECTOR_DB = Chroma.from_documents(chunks, embeddings)
        else:
            VECTOR_DB.add_documents(chunks)

    return {
        "chunks": len(chunks),
        "indexed_files": list(UPLOADED_FILES),
        "skipped_duplicates": skipped
    }


# ── Rule-based smalltalk reply (no LLM needed) ──────────────────────────────
def rule_based_smalltalk(text: str):
    """
    Returns a reply string if the text is obvious smalltalk, else None.
    Checked BEFORE calling the LLM classifier.
    """
    t = text.lower().strip()

    # Name introduction
    name_match = re.search(
        r"(?:^|\b)(?:im|i am|i'm|my name is|call me)\s+([a-zA-Z]+)", t
    )
    if name_match:
        name = name_match.group(1).capitalize()
        return f"Nice to meet you, **{name}**! 😊 How can I help you today?"

    # Farewell
    if re.search(r"\b(bye|goodbye|see you|cya|take care)\b", t):
        return "Goodbye! 👋 Have a great day!"

    # Thanks
    if re.search(r"\b(thanks|thank you|thx|thank u)\b", t):
        return "You're welcome! 😊 Let me know if you need anything else."

    # Pure greetings — only if the ENTIRE message is a greeting phrase
    # (avoids catching "hello what is ai")
    if re.fullmatch(
        r"(hi+|hello+|hey+|hii+|helo|hai|howdy|good\s*(morning|afternoon|evening|day)|"
        r"what'?s\s*up|wassup|sup)[!?.]*",
        t
    ):
        return "Hello! 😊 How can I assist you today?"

    return None  # not smalltalk — proceed to LLM classifier


# ── LLM classifier (only called for ambiguous inputs) ───────────────────────
def llm_classify(llm: ChatGroq, user_text: str) -> str:
    """Returns 'SMALLTALK' or 'QUESTION'."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a classifier for a Student Handbook chatbot.
Classify the user message as SMALLTALK or QUESTION.

SMALLTALK = ONLY:
  - Pure greetings with no question: hi, hello, hey
  - Name introductions: "im X", "my name is X"
  - Thanks, farewell with no question attached

QUESTION = EVERYTHING else, including:
  - General knowledge: "what is AI", "explain ML", "how does X work"
  - Document questions: policies, fees, rules, subjects
  - Any sentence with what/why/how/when/where/who/explain/define/is/are/can/does
  - Typo-filled questions: "what s ai", "wht is python", "explan ml"

When in doubt → QUESTION.

Reply with exactly one word: SMALLTALK or QUESTION"""),
        ("human", "{input}")
    ])
    try:
        result = (prompt | llm | StrOutputParser()).invoke({"input": user_text}).strip().upper()
        if "SMALLTALK" in result and "QUESTION" not in result:
            return "SMALLTALK"
        return "QUESTION"
    except Exception:
        return "QUESTION"  # fail safe — always attempt RAG on error


# ── Combined intent classifier ───────────────────────────────────────────────
def classify_intent(llm: ChatGroq, user_text: str):
    """
    Returns (intent, reply):
      ("greeting", "Hello!...")  for smalltalk
      ("question", None)         for document/general questions
    """
    # Step 1: fast rule-based check (no LLM cost)
    reply = rule_based_smalltalk(user_text)
    if reply:
        return "greeting", reply

    # Step 2: LLM for ambiguous cases
    label = llm_classify(llm, user_text)
    if label == "SMALLTALK":
        return "greeting", "Hello! 😊 How can I assist you today?"

    return "question", None


@app.post("/ask")
def ask(data: Question):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=data.model_name
    )

    # Intent check FIRST — greetings work even without uploaded documents
    intent, greeting_response = classify_intent(llm, data.question)
    if intent == "greeting":
        return {"answer": greeting_response, "sources": []}

    # Only block document questions if no docs uploaded yet
    if VECTOR_DB is None:
        return {
            "answer": "❗ Please upload a document from the sidebar first, then ask your question!",
            "sources": []
        }

    # RAG pipeline
    retriever = VECTOR_DB.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are a precise RAG assistant for a Student Handbook.
Answer ONLY from the provided context below.
If the answer is not found in the context, say exactly: "I could not find this in the document."
Do not use any outside knowledge.

Context:
{context}

Question:
{question}
""")

    retrieved_docs = retriever.invoke(data.question)
    context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
    sources = [
        {"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", 0)}
        for doc in retrieved_docs
    ]

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": data.question})

    return {"answer": answer, "sources": sources}
