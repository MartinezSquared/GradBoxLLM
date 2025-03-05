import os
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv
import nest_asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import retrieve_text_chunks, load_vectorstore

# Allow nested asyncio loops
nest_asyncio.apply()
load_dotenv()

# Load Auth Config
with open("./cloud_app/.streamlit/auth_streamlit_app_lite.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Streamlit UI
st.title("GradBoxLLM - Textbook AI Assistant Demo")
st.markdown(
    "[Like what you see? Star the GitHub Project!](https://github.com/MartinezSquared/GradBoxLLM)",
    unsafe_allow_html=True
)

# Authentication
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

# Retrieve secrets
HF_TOKEN = st.secrets.get("HF_TOKEN")
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY")

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
                faiss_path = "./cloud_app/faissIndex"
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
            
            if not GEMINI_API_KEY:
                st.error("No Google API key found.")
            else:
                with st.spinner("Generating response..."):
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-thinking-exp-01-21",
                        temperature=0,
                        max_tokens=None,
                        api_key=GEMINI_API_KEY
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

