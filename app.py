import warnings
import logging
import os
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  
import pypdf

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

# ----------------- Show previous chat -----------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ----------------- Cached Embeddings ----  -------------
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
                documents.append(
                    Document(
                        page_content=page.extract_text(),
                        metadata={"source": file.name, "page": i}
                    )
                )
        elif suffix == "docx":
            loader = Docx2txtLoader(file)
            for page_content in loader.load():
                documents.append(Document(
                    page_content=page_content, 
                    metadata={"source": file.name}))
        else:
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return splitter.split_documents(documents)

# ----------------- Sidebar -----------------
st.sidebar.title("⚙ RAG Settings")
selected_model = st.sidebar.selectbox(
    "Select LLM",
    [   "meta-llama/llama-4-scout-17b-16e-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "openai/gpt-oss-safeguard-20b", 
        "qwen/qwen3-32b"]
)
st.sidebar.write("Selected Model:", selected_model)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs / DOCX",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# ----------------- Process new files -----------------
new_files = []
for file in uploaded_files:
    if file.name not in st.session_state.uploaded_files:
        new_files.append(file)
        st.session_state.uploaded_files.append(file.name)

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

    st.sidebar.success("✅ Documents indexed successfully")

# ----------------- Chat Input -----------------
query = st.chat_input("Ask a question from the documents...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    if st.session_state.vectorstore is None:
        response = "❗ Please upload documents first."
    else:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template("""
You are a precise RAG assistant.
Answer ONLY from the provided context.
If not found, say:
"I could not find this in the document."

Context:
{context}

Question:
{question}
""")

        llm = ChatGroq(
            model_name=selected_model
        )

        rag_chain = (
            {
                "context": retriever
                | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(query)

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
