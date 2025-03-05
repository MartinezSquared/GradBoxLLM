import streamlit as st
import nest_asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import retrieve_text_chunks, load_vectorstore

# Allow nested asyncio loops
nest_asyncio.apply()

# Retrieve Hugging Face and Gemini keys from Streamlit Secrets
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY")

st.title("GradBoxLLM - Textbook AI Assistant Demo")

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

