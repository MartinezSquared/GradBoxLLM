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
import random
from PIL import Image

# Import functions from your modules
from pdf_to_vectorstore import load_pdf_pages, split_pdf_pages, compute_pdf_embeddings, create_faiss_index
from rag_to_gemini import retrieve_text_chunks, load_vectorstore, get_faiss_index_path 

import yaml
from yaml.loader import SafeLoader

st.set_page_config(
    page_title="GradBox",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration from YAML file
with open('./.streamlit/login.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Ensure the cookie key is provided; if not, stop execution.
if not config.get('cookie', {}).get('key'):
    st.error("Cookie key not set in YAML config. Please update '.streamlit/login.yaml'.")
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
FAISS_INDEX_PATH = get_faiss_index_path("./tmp/vectorstore")

def build_vectorstore(pdf_files):
    """Process uploaded PDFs, build a vectorstore, and save it locally."""
    all_chunks = []
    if len(pdf_files) > 1:
        pdf_status = f"Reading {len(pdf_files)} PDF Files :nerd_face:"
    else:
        pdf_status = f"Reading the PDF File :nerd_face:"

    with st.spinner(pdf_status):
        for pdf_file in pdf_files:
            # Save the uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name
            pages = asyncio.run(load_pdf_pages(tmp_path))
            chunks = split_pdf_pages(pages, chunk_size=200, chunk_overlap=50)
            all_chunks.extend(chunks)
            os.remove(tmp_path)
    
    with st.spinner(f"Embedding {len(all_chunks)} chunks into vectors :exploding_head:"):
        # Compute embeddings (use "cuda" if available; otherwise "cpu")
        embeddings = compute_pdf_embeddings(all_chunks, device="cuda")
        # Create the FAISS vectorstore
        index = create_faiss_index(embeddings, all_chunks)
        # Save the index locally
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        index.save_local(FAISS_INDEX_PATH)
    return index


# --- Streamlit UI ---
st.title(":books: GradBox - Textbook RAG")
st.markdown("---")

# Create a container for the login widget.
login_container = st.empty()
with login_container.container():
    authenticator.login(
        "main",
        fields={
            "Form name": "Login",
            "Username": "Username",
            "Password": "Password",
            "Login": "Login",
            "Captcha": "Captcha"
        },
        key="login"
    )

# Check authentication state via st.session_state
if st.session_state.get("authentication_status"):
    # Remove the login widget by clearing the container.
    login_container.empty()
    name = st.session_state.get("name")
    username = st.session_state.get("username")
    st.sidebar.success(f"Welcome, {name}!")
    # Render a logout button with a unique key.
    authenticator.logout("Logout", "sidebar", key="logout-widget")

    # --- Vectorstore Management (Sidebar) ---
    st.sidebar.header("Vectorstore Management")
    uploaded_files = st.sidebar.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
    if st.sidebar.button("Build/Update Vectorstore") and uploaded_files:
        index = build_vectorstore(uploaded_files)
        st.session_state.index = index
        st.sidebar.success(f"Loaded {FAISS_INDEX_PATH}")
        st.success(f"Loaded {FAISS_INDEX_PATH}")

    if st.sidebar.button("Delete Vectorstore"):
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        st.session_state.index = None
        st.sidebar.success(f"{FAISS_INDEX_PATH} deleted.")
        st.success(f"{FAISS_INDEX_PATH} deleted.")

    # Check if vectorstore is loaded; if not, load from disk
    if "index" not in st.session_state or st.session_state.index is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            st.info("Please upload PDFs and build a vectorstore.")
            st.sidebar.info("Please upload PDFs and build a vectorstore.")
        else:
            st.session_state.index = index
            st.sidebar.success(f"{FAISS_INDEX_PATH} loaded.")
            index = load_vectorstore()

    # --- Main Content: Query using Gemini ---
    st.header("Ask a Question")
    user_query = st.text_input("Enter your query here")
    if st.button("Submit Query") and user_query:
        if "index" not in st.session_state or st.session_state.index is None:
            st.error("No vectorstore available. Please build the vectorstore first.")
        else:
            index = st.session_state.index
            st.markdown(
                "[Like what you see? Star the GitHub Project!](https://github.com/MartinezSquared/GradBoxLLM)",
                unsafe_allow_html=True
            )
            retrieved_chunks = retrieve_text_chunks(user_query, st.session_state["index"], k=4)
            
            formatted_chunks = []
            for i, chunk in enumerate(retrieved_chunks, start=1):
                title = chunk.metadata.get("title", "Unknown Title")
                page = chunk.metadata.get("page", "Unknown Page")
                formatted_chunk = f"Chunk {i}:\nTitle: {title}\nPage: {page}\n{chunk.page_content}\n---\n"
                formatted_chunks.append(formatted_chunk)
            
            retrieved_text = "\n".join(formatted_chunks)
            
            GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
            if not GEMINI_API_KEY:
                st.error("No Google API key found.")
            else:
                with st.spinner("Generating response... :robot_face:"):
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-thinking-exp-01-21",
                        temperature=0,
                        max_tokens=None,
                        api_key=GEMINI_API_KEY
                    )
                    prompt = (f"""
System:
You are a helpful assistant that answers the user's question. 
First come up with an answer then review the chunks of text from RAG.
Use the most relavent chunks to support your answer
Provide the textbook metadata as a reference if you thought it was relavent.
Otherwise, if none of the RAG text chunks were relavent, 
answer the question using advance reasoning and at the end tell the user that none of the RAG chunks were used due to relavance.

Context:
{retrieved_text}

User:
{user_query}

Answer:

""")
                    response = llm.invoke(prompt)
                st.subheader("Gemini's Response")
                st.write(response.content)
               
                st.subheader("Retrieved Chunks")
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    title = chunk.metadata.get("title", "Unknown Title")
                    page = chunk.metadata.get("page", "Unknown Page")
                    st.markdown("---")
                    st.markdown(f"### Chunk {i}")
                    st.markdown(f"**Title:** {title}")
                    st.markdown(f"**Page:** {page}")
                    st.write(chunk.page_content)
                
                st.subheader("Prompt to Gemini")
                st.code(prompt)
elif st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")

