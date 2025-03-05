import os
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv
import nest_asyncio
import hashlib
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import retrieve_text_chunks, load_vectorstore

# Allow nested asyncio loops
nest_asyncio.apply()
load_dotenv()

# Load Auth Config
with open("./public_app/.streamlit/auth_streamlit_app_lite.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Streamlit UI
st.title("GradBoxLLM - Textbook AI Assistant")

# Authentication
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

# User API Key Input
def hash_key(key):
    return hashlib.sha256(key.encode()).hexdigest() if key else None

st.sidebar.header("API Key Configuration")
user_gemini_key = st.sidebar.text_input("Enter your Google API Key", type="password")
user_hf_token = st.sidebar.text_input("Enter your Hugging Face Token", type="password")

hashed_gemini_key = hash_key(user_gemini_key)
hashed_hf_token = hash_key(user_hf_token)

if user_gemini_key:
    st.session_state["GEMINI_API_KEY"] = user_gemini_key
if user_hf_token:
    st.session_state["HF_TOKEN"] = user_hf_token

# Authentication UI
login_container = st.empty()
with login_container.container():
    authenticator.login(
        "main",
        fields={"Form name": "Login", "Username": "Username", "Password": "Password", "Login": "Login"},
        key="login"
    )

if st.session_state.get("authentication_status"):
    login_container.empty()
    name = st.session_state.get("name")
    st.success(f"Welcome, {name}!")
    authenticator.logout("Logout", "main", key="logout-widget")

    # --- Load FAISS Index ---
    if "index" not in st.session_state:
        if st.button("Load Vectorstore"):
            with st.spinner("Loading vectorstore..."):
                faiss_path = "./public_app/faissIndex"
                index = load_vectorstore(faiss_path)
                if index:
                    st.session_state["index"] = index
                    st.success("Vectorstore loaded successfully.")
                else:
                    st.error("No FAISS index found.")

    if "index" in st.session_state:
        st.header("Ask a Nursing Related Question")
        user_query = st.text_input("Enter your query here")
        if st.button("Submit Query") and user_query:
            retrieved_chunks = retrieve_text_chunks(user_query, st.session_state["index"], k=4)
            
            formatted_chunks = []
            for i, chunk in enumerate(retrieved_chunks, start=1):
                title = chunk.metadata.get("title", "Unknown Title")
                page = chunk.metadata.get("page", "Unknown Page")
                formatted_chunk = f"Chunk {i}:\nTitle: {title}\nPage: {page}\n{chunk.page_content}\n---\n"
                formatted_chunks.append(formatted_chunk)
            
            retrieved_text = "\n".join(formatted_chunks)
            
            if "GEMINI_API_KEY" not in st.session_state:
                st.error("No Google API key found. Please enter it in the sidebar.")
            else:
                with st.spinner("Generating response..."):
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-thinking-exp-01-21",
                        temperature=0,
                        max_tokens=None,
                        api_key=st.session_state["GEMINI_API_KEY"]
                    )
                    response = llm.invoke(f"""
System:
You are a helpful assistant using textbook knowledge.
Context:
{retrieved_text}
Question:
{user_query}
Answer:
                    """)
                
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
                st.code(f"""
System:
You are a helpful assistant using textbook knowledge.
Context:
{retrieved_text}
Question:
{user_query}
Answer:
                """)

