import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

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

def main():
## Setup
    # Load API Keys
    load_dotenv()
    
## Streamlit UI
    # Chat Header
    st.set_page_config(page_icon=":books:", page_title="GradBoxLLM")
    st.header(":books: GradBoxLLM - Your Textbook AI Assistant")
    
    # User Prompting
    st.text_input("Ask a question about the textbook")

    # Uploading Documents
    with st.sidebar:
        st.subheader("Your Documents")
        pdfFiles = st.file_uploader(
            "Upload Textbook and Click Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract PDF text
                extractedText = extractPdf(pdfFiles)
                # st.write(extractedText)
                
                # Chunk PDF Text
                chunkedText = chunkText(extractedText)
                #st.write(chunkedText)
                
                # Encode Chunk into Vector Store
                vectorStore = getVectorStore(chunkedText)

                # Chain Conversation
                conversation = getConversation(vectorStore)
                

if __name__ == "__main__":
    main()