import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from dotenv import load_dotenv
import nest_asyncio

# --- Local imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import retrieve_text_chunks, load_vectorstore

# Allow nested asyncio loops
nest_asyncio.apply()

# Load environment variables (optional)
load_dotenv()

# Load the Streamlit Authenticator config file
with open("./.streamlit/auth_streamlit_app_lite.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Ensure we have a valid cookie key
if not config.get("cookie", {}).get("key"):
    st.error("Cookie key not set in YAML config. Please update '.streamlit/config.yaml'.")
    st.stop()
else:
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"]
    )

# Retrieve API keys from Streamlit secrets
HUGGINGFACE_API_KEY = st.secrets.get("HF_TOKEN")
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY")

st.title("GradBoxLLM - Textbook AI Assistant Demo")

# Render the login widget
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

# Check authentication status
if st.session_state.get("authentication_status"):
    # User is authenticated
    name = st.session_state.get("name")
    st.success(f"Welcome, {name}!")
    authenticator.logout("Logout", "main")

    # --- Main App Logic (from gradapp.py) ---
    # Button to load heavy resources on demand
    if "index" not in st.session_state:
        if st.button("Load Vectorstore"):
            with st.spinner("Loading vectorstore..."):
                index = load_vectorstore("./cloud_app/faissIndex")
                if index is None:
                    st.error("No vectorstore found. Please build the vectorstore offline and place it at './faissIndex'.")
                else:
                    st.session_state["index"] = index
                    st.success("Vectorstore loaded successfully.")
    else:
        st.success("Vectorstore is already loaded.")

    # Only show query interface if the vectorstore is loaded
    if st.session_state.get("index") is not None:
        st.header("Ask a Question")
        user_query = st.text_input("Enter your query here")
        if st.button("Submit Query") and user_query:
            index = st.session_state["index"]
            retrieved_chunks = retrieve_text_chunks(user_query, index, k=4)
            retrieved_text = "".join([chunk.page_content + "\n" for chunk in retrieved_chunks])

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

            if not GEMINI_API_KEY:
                st.error("No Gemini API key found in Streamlit secrets.")
            else:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-thinking-exp-01-21",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key=GEMINI_API_KEY
                )
                response = llm.invoke(prompt)
                st.subheader("Gemini's Response")
                st.write(response.content)

elif st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")

else:
    st.warning("Please enter your username and password")

