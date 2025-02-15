import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA
from htmlTemplates import css, bot_template, user_template
from InstructorEmbedding import INSTRUCTOR
import faiss
import numpy as np


class InstructorEmbeddingsWrapper:
    def __init__(self):
        self.model = INSTRUCTOR('hkunlp/instructor-large')

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: # Changed return type to list[np.ndarray]
        instruction = "Represent the document for retrieval:"
        embeddings_list = []
        for text in texts:
            embedding = self.model.encode([[instruction, text]])
            embeddings_list.append(embedding.flatten()) # Flatten but keep as numpy array
        return embeddings_list

    def embed_query(self, text: str) -> list[float]:
        instruction = "Represent the query for retrieval:"
        embedding = self.model.encode([[instruction, text]])
        return embedding.flatten().tolist()


def getTextFromPDF(pdfDocs):
    text = ""
    for pdf in pdfDocs:
        pdfReader = PdfReader(pdf)
        for page in pdfReader.pages:
            text += page.extract_text()
    return text

def getTextChunks(text):
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len
        )
    chunks = textSplitter.split_text(text=text)
    return chunks

def getVectorStore(chunkedText):
    instructor_embeddings_obj = InstructorEmbeddingsWrapper()
    embedding_dimension = 768
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}

    vectorStore = FAISS(
        embedding_function=instructor_embeddings_obj.embed_documents,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # Explicitly calculate and convert embeddings to NumPy array within add_texts
    embeddings = instructor_embeddings_obj.embed_documents(chunkedText) # Calculate embeddings using wrapper
    embeddings_np = np.array(embeddings, dtype=np.float32) # Convert to NumPy array, ensure dtype

    vectorStore.add_embeddings(texts=chunkedText, embeddings=embeddings_np) # Use add_embeddings with pre-calculated, converted embeddings
    return vectorStore

def getQuestionAnswerChain(vectorStore):
    llm = CTransformers(model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                                model_type="mistral",
                                gpu_layers=0,
                                temperature=0.5)
    qaChain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectorStore.as_retriever())
    return qaChain

def handle_userinput(userQuestion):
    if st.session_state.qaChain is not None:
        response = st.session_state.qaChain({'query': userQuestion})
        answer = response['result']

        st.write(user_template.replace("{{MSG}}", userQuestion), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
    else:
        st.error("Please upload documents and process them to initialize the QA Chain.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Multi PDF Chatbot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversationHistory" not in st.session_state:
        st.session_state.conversationHistory = None
    if "qaChain" not in st.session_state:
        st.session_state.qaChain = None

    st.header("Multi PDF Chatbot :books:")
    userQuestion = st.text_input("Ask questions about your PDF documents:")
    if userQuestion:
        handle_userinput(userQuestion)

    with st.sidebar:
        st.subheader("Your documents")
        pdfDocs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                rawText = getTextFromPDF(pdfDocs)

                # Get text chunks
                chunkedText = getTextChunks(rawText)

                # Create vector store
                vectorStore = getVectorStore(chunkedText)

                # Create question-answering chain
                qaChain = getQuestionAnswerChain(vectorStore)
                st.session_state.qaChain = qaChain

                st.success("Done")

if __name__ == '__main__':
    main()




