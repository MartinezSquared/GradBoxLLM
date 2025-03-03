import streamlit as st
import streamlit_authenticator as stauth
import os
import nest_asyncio
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_to_gemini import retrieve_text_chunks
from langchain_community.vectorstores import FAISS

import yaml
from yaml.loader import SafeLoader

with open('./.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

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

# Constants
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
        index = FAISS.load_local(FAISS_INDEX_PATH, embed_model, allow_dangerous_deserialization=True)
        return index
    else:
        return None

# --- Streamlit UI ---

st.title("GradBoxLLM - Textbook AI Assistant Demo")

# Load the vectorstore (built offline)
index = load_vectorstore()
if index is None:
    st.error(f"No vectorstore found. Please build the vectorstore offline and place it at '{FAISS_INDEX_PATH}'.")
else:
    st.success("Vectorstore loaded from disk.")

st.header("Ask a Question")
user_query = st.text_input("Enter your query here")

if st.button("Submit Query") and user_query:
    if index is None:
        st.error("No vectorstore available. Please build the vectorstore offline.")
    else:
        # Retrieve relevant text chunks from the vectorstore
        retrieved_chunks = retrieve_text_chunks(user_query, index, k=4)
        retrieved_text = ""
        for chunk in retrieved_chunks:
            retrieved_text += chunk.page_content + "\n"
        
        # Compose the prompt for Gemini
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
        
        # Initialize Gemini LLM using the API key from environment variables
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

