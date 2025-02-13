import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

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
        chunk_size = 3000,
        chunk_overlap = 1000,
        length_function = len
    )
    chunks = textSplitter.split_text(text)
    return chunks

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
                st.write(chunkedText)
                
                # Encode Chunk into Vector Store
                
                

if __name__ == "__main__":
    main()