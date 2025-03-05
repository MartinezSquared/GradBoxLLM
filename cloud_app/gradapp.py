import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from dotenv import load_dotenv
import nest_asyncio

# -- Local imports --
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import retrieve_text_chunks, load_vectorstore

# Allow nested asyncio loops
nest_asyncio.apply()
load_dotenv()

# Load Auth Config
with open("./cloud_app/.streamlit/auth_streamlit_app_lite.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Instantiate the Authenticator
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

# Retrieve secrets
HF_TOKEN = st.secrets.get("HF_TOKEN")
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Streamlit UI
st.title("GradBoxLLM - Textbook AI Assistant Demo")

# Here, capture the return values of the login widget:
name, authentication_status, username = authenticator.login("main")

# Check the login status
if authentication_status is None:
    st.warning("Please enter your username and password")
elif authentication_status is False:
    st.error("Username/password is incorrect")
else:
    # User is authenticated
    st.success(f"Welcome, {name}!")
    authenticator.logout("Logout", "sidebar")

    # --- Main App Logic ---
    if "index" not in st.session_state:
        if st.button("Load Vectorstore"):
            with st.spinner("Loading vectorstore..."):
                index = load_vectorstore("./cloud_app/faissIndex")
                if index is None:
                    st.error("No vectorstore found. Please build it offline and place at './faissIndex'.")
                else:
                    st.session_state["index"] = index
                    st.success("Vectorstore loaded successfully.")
    else:
        st.success("Vectorstore is already loaded.")

    if st.session_state.get("index") is not None:
        st.header("Ask a Question")
        user_query = st.text_input("Enter your query here")
        if st.button("Submit Query") and user_query:
            index = st.session_state["index"]
            retrieved_chunks = retrieve_text_chunks(user_query, index, k=4)
            retrieved_text = "".join(chunk.page_content + "\n" for chunk in retrieved_chunks)

            # Compose prompt
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
                st.error("No Google API key (Gemini) found in secrets.")
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

