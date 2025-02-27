




chat = ChatVertexAI(model="gemini-pro", google_api_key="...")


def extractPdf(pdfFiles):
    text = ""
    # For each PDF, Read the Page, and Extract to one object
    for pdf in pdfFiles:
        pyPdf = PdfReader(pdf)
        for page in pyPdf.pages:
            text += page.extract_text()
    return text


def chunkText(text):
    textSplitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 300,
        chunk_overlap = 150,
        length_function = len
    )
    chunks = textSplitter.split_text(text)
    return chunks

def getVectorStore(chunkedText):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorStore = FAISS.from_texts(texts=chunkedText, embedding = embeddings)
    return vectorStore

def getConversation(vectorStore, llm):
    contextualizeQuestionSysPrompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is.")
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), contextualize_q_prompt)
    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # Chain to answer questions based on retrieved documents, considering chat history.
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    # Final RAG chain combining history-aware retriever and question-answering chain.
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)
    # Wraps the RAG chain to manage chat history statefully.
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state:
            st.session_state[session_id] = ChatMessageHistory()
        return st.session_state[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversational_rag_chain.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        st.session_state.chat_history = st.session_state[st.session_state.session_id].messages
        for i, message in enumerate(st.session_state.chat_history):
            if isinstance(message, HumanMessage):
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.write("Please process documents first!")

def main():
## Setup
    # Load API Keys
    load_dotenv()
    
## Streamlit UI
    # Chat Header
    st.set_page_config(page_icon=":books:", page_title="GradBoxLLM")
    st.header(":books: GradBoxLLM - Your Textbook AI Assistant")
    
    # User Prompting
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = extractPdf(pdf_docs)
                chunkedText = chunkText(raw_text)
                vectorStore = getVectorStore(chunkedText)
                # Initialize ChatVertexAI with Gemini 1.5 Flash model
                llm = ChatVertexAI(model="gemini-1.5-flash")
                st.session_state.conversational_rag_chain = getConversation(vectorStore, llm)
                st.success("Done")
                

if __name__ == "__main__":
    main()
