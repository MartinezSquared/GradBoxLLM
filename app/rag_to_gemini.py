import streamlit as st
import streamlit_authenticator as stauth
import os
import asyncio
import nest_asyncio
import tempfile
import shutil
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# Import functions from your modules
from pdf_to_vectorstore import load_pdf_pages, split_pdf_pages, compute_pdf_embeddings, create_faiss_index
from rag_to_gemini import retrieve_text_chunks

import yaml
from yaml.loader import SafeLoader

# Load configuration from YAML file
with open('./.streamlit/auth_streamlit_app_lite.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Ensure the cookie key is provided; if not, stop execution.
if not config.get('cookie', {}).get('key'):
    st.error("Cookie key not set in YAML config. Please update '.streamlit/auth_streamlit_app_lite.yaml'.")
    st.stop()
else:
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

# Allow nested asyncio loops
nest_asyncio.apply()

# Load environment variables (including GOOGLE_API_KEY)
load_dotenv()

# Constants for vectorstore path
FAISS_INDEX_PATH = "./faissIndex"

def load_vectorstore():
    """
    Loads the FAISS vectorstore from disk using a CPU-based SentenceTransformer.
    """
    embed_model = SentenceTransformer(
        "Lajavaness/bilingual-embedding-small",
        trust_remote_code=True,
        device="cpu"
    )
    if os.path.exists(FAISS_INDEX_PATH):
        from langchain_community.vectorstores import FAISS  # local import for clarity
        index = FAISS.load_local(FAISS_INDEX_PATH, embed_model, allow_dangerous_deserialization=True)
        return index
    else:
        return None

def build_vectorstore(pdf_files):
    """Process uploaded PDFs, build a vectorstore, and save it locally."""
    all_chunks = []
    for pdf_file in pdf_files:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        st.write(f"Processing: {tmp_path}")
        pages = asyncio.run(load_pdf_pages(tmp_path))
        chunks = split_pdf_pages(pages, chunk_size=200, chunk_overlap=50)
        all_chunks.extend(chunks)
        os.remove(tmp_path)
    st.write(f"Total number of chunks: {len(all_chunks)}")

    # Compute embeddings (use "cuda" if available; otherwise "cpu")
    embeddings = compute_pdf_embeddings(all_chunks, device="cuda")

    # Create the FAISS vectorstore
    index = create_faiss_index(embeddings, all_chunks)

    # Save the index locally
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    index.save_local(FAISS_INDEX_PATH)
    return index

# --- Streamlit UI ---

st.title("GradBoxLLM - Textbook AI Assistant")

# Render the login widget using the new syntax.
authenticator.login(
    "main",
    fields={
        "Form name": "Login",
        "Username": "Username",
        "Password": "Password",
        "Login": "Login",
        "Captcha": "Captcha"
    }
)

# Check the authentication status from session state.
if st.session_state.get("authentication_status"):
    name = st.session_state.get("name")
    st.success(f"Welcome, {name}!")
    # Render a logout button with a unique key.
    authenticator.logout("Logout", "main", key="logout-widget")

    # --- Vectorstore Management (Sidebar) ---
    st.sidebar.header("Vectorstore Management")
    uploaded_files = st.sidebar.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
    if st.sidebar.button("Build/Update Vectorstore") and uploaded_files:
        index = build_vectorstore(uploaded_files)
        st.session_state.index = index
        st.sidebar.success("Vectorstore built and saved.")

    if st.sidebar.button("Delete Vectorstore"):
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        st.session_state.index = None
        st.sidebar.success("Vectorstore deleted.")

    # Check if vectorstore is loaded; if not, load from disk
    if "index" not in st.session_state or st.session_state.index is None:
        index = load_vectorstore()
        if index is not None:
            st.session_state.index = index
            st.sidebar.success("Vectorstore loaded from disk.")
        else:
            st.sidebar.warning("No vectorstore found. Please upload PDFs to build one.")
            st.info("Vectorstore not available. Please use the sidebar to upload PDFs and build the vectorstore.")

    # --- Main Content: Query using Gemini ---
    st.header("Ask a Question")
    user_query = st.text_input("Enter your query here")
    if st.button("Submit Query") and user_query:
        if "index" not in st.session_state or st.session_state.index is None:
            st.error("No vectorstore available. Please build the vectorstore first.")
        else:
            index = st.session_state.index
            retrieved_chunks = retrieve_text_chunks(user_query, index, k=4)
            retrieved_text = ""
            for chunk in retrieved_chunks:
                retrieved_text += chunk.page_content + "\n"
            # Compose the prompt for Gemini.
            prompt = f"""
System:
You are a helpful nursing assistant that uses relevant context from a textbook and advanced reasoning to answer the user's question.
Context:
{retrieved_text}
Question:
{user_query}
Answer:
            """
            st.subheader("Prompt to Gemini")
            st.code(prompt)

            # Initialize Gemini LLM using the API key from environment variables.
            GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
            if GOOGLE_API_KEY is None:
                st.error("GOOGLE_API_KEY not set in environment.")
            else:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-thinking-exp-01-21",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key=GOOGLE_API_KEY
                )
                response = llm.invoke(prompt)
                st.subheader("Gemini's Response")
                st.write(response.content)

elif st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password")

