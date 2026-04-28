import streamlit as st
import requests

API_URL = "http://localhost:8000"

MODELS = [
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "openai/gpt-oss-safeguard-20b",
]

# ✅ Fix 3: MIME type mapping for PDF and DOCX
MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📄 Student HandBook PDF Q/A Chatbot")

# ✅ Fix 3: Session state for chat history + welcome message (was missing entirely)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if not st.session_state.greeted:
    welcome = "Hello! 👋 I'm here to help you with your Student Handbook. Upload PDFs/DOCX and ask any student-related question!"
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    st.session_state.greeted = True

# --- Sidebar: Upload + Model Selection ---
st.sidebar.title("⚙ RAG Settings")

model = st.sidebar.selectbox("Choose a model", MODELS)

st.sidebar.subheader("Upload Documents")

# ✅ Fix 1: Accept both pdf and docx (was only pdf)
files = st.sidebar.file_uploader(
    "Upload PDFs / DOCX",
    type=["pdf", "docx"],      # ✅ Fix 1
    accept_multiple_files=True
)

if st.sidebar.button("Upload Documents"):
    if files:
        new_files = [f for f in files if f.name not in st.session_state.uploaded_files]
        if not new_files:
            st.sidebar.warning("All selected files have already been uploaded.")
        else:
            with st.spinner("Uploading and indexing..."):
                try:
                    # ✅ Fix 2: Use correct MIME type per file extension
                    file_tuples = []
                    for f in new_files:
                        ext = f.name.split(".")[-1].lower()
                        mime = MIME_TYPES.get(ext, "application/octet-stream")
                        file_tuples.append(("files", (f.name, f, mime)))

                    response = requests.post(f"{API_URL}/upload", files=file_tuples)
                    response.raise_for_status()
                    data = response.json()

                    for f in new_files:
                        st.session_state.uploaded_files.append(f.name)

                    msg = f"✅ Indexed **{data['chunks']}** chunks from **{len(new_files)}** file(s)."
                    if data.get("skipped_duplicates"):
                        msg += f"\n\n⚠️ Skipped duplicates: {', '.join(data['skipped_duplicates'])}"
                    st.sidebar.success(msg)

                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"Upload failed: {e}")
    else:
        st.sidebar.warning("Please select at least one file.")

# --- Show uploaded files list ---
if st.session_state.uploaded_files:
    with st.sidebar.expander("📁 Indexed Files"):
        for name in st.session_state.uploaded_files:
            st.write(f"- {name}")

# --- Chat history display ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# --- Chat input ---
question = st.chat_input("Say hi or ask a question from the documents...")

if question:
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "model_name": model}
            )
            response.raise_for_status()
            data = response.json()

            answer = data["answer"]
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except requests.exceptions.RequestException as e:
            err = f"❗ Request failed: {e}"
            st.chat_message("assistant").markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
