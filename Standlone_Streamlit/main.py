import warnings
import logging
import os
import io
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import pypdf
import tempfile

# ----------------- Setup -----------------
load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📄 Student HandBook PDF Q/A Chatbot")

# ----------------- Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ✅ Welcome message shown once
if "greeted" not in st.session_state:
    st.session_state.greeted = False

if not st.session_state.greeted:
    welcome = "Hello! 👋 I'm here to help you with your Student Handbook. Upload PDFs/DOCX and ask any student-related question!"
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    st.session_state.greeted = True

# ----------------- Show previous chat -----------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ----------------- Cached Embeddings -----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ----------------- Process files -----------------
def process_files(files):
    documents = []

    for file in files:
        suffix = file.name.split(".")[-1].lower()

        if suffix == "pdf":
            pdf = pypdf.PdfReader(io.BytesIO(file.read()))
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": file.name, "page": i}
                        )
                    )

        elif suffix == "docx":
            # ✅ Fix: write to a real temp file path (Docx2txtLoader needs a path, not a buffer)
            content = file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                loader = Docx2txtLoader(tmp_path)
                for doc in loader.load():
                    doc.metadata["source"] = file.name
                    documents.append(doc)
            finally:
                os.unlink(tmp_path)

        else:
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return splitter.split_documents(documents)


# ----------------- Greeting Handler (Rule-based) -----------------
def rule_based_smalltalk(text: str):
    """Returns a reply string if the text is obvious smalltalk, else None."""
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
    if re.fullmatch(
        r"(hi+|hello+|hey+|hii+|helo|hai|howdy|good\s*(morning|afternoon|evening|day)|"
        r"what'?s\s*up|wassup|sup)[!?.]*",
        t
    ):
        return "Hello! 😊 How can I assist you today?"

    return None  # not smalltalk — proceed to LLM classifier


# ----------------- Greeting Handler (LLM-based fallback) -----------------
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


# ----------------- Sidebar -----------------
st.sidebar.title("⚙ RAG Settings")

selected_model = st.sidebar.selectbox(
    "Select LLM",
    [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3-32b"
    ]
)
st.sidebar.write("Selected Model:", selected_model)

# ✅ Fix: Accept both pdf and docx
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs / DOCX",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# ----------------- Process new files only (skip duplicates) -----------------
new_files = []
for file in uploaded_files:
    if file.name not in st.session_state.uploaded_files:
        new_files.append(file)

if new_files:
    with st.spinner("⚡ Processing documents..."):
        new_chunks = process_files(new_files)

        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma.from_documents(
                documents=new_chunks,
                embedding=load_embeddings()
            )
        else:
            st.session_state.vectorstore.add_documents(new_chunks)

        for f in new_files:
            st.session_state.uploaded_files.append(f.name)

    st.sidebar.success(f"✅ Indexed **{len(new_chunks)}** chunks from **{len(new_files)}** file(s).")

# ✅ Show list of already-indexed files
if st.session_state.uploaded_files:
    with st.sidebar.expander("📁 Indexed Files"):
        for name in st.session_state.uploaded_files:
            st.write(f"- {name}")

# ----------------- Chat Input -----------------
query = st.chat_input("Say hi or ask a question from the documents...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    llm = ChatGroq(model_name=selected_model)

    # ✅ Intent check FIRST — greetings work even without uploaded documents
    intent, greeting_reply = classify_intent(llm, query)

    if intent == "greeting":
        response = greeting_reply
        sources = []

    elif st.session_state.vectorstore is None:
        response = "❗ Please upload a document from the sidebar first, then ask your question!"
        sources = []

    else:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

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

        retrieved_docs = retriever.invoke(query)
        context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
        sources = [
            {"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", 0)}
            for doc in retrieved_docs
        ]

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context_str, "question": query})

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
