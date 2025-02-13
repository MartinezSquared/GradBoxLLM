import streamlit as st

def main():
    # Chat Header
    st.set_page_config(page_icon=":books:", page_title="GradBoxLLM")
    st.header(":books: GradBoxLLM - Your Textbook AI Assistant")
    
    # User Prompting
    st.text_input("Ask a question about the textbook")

    # Uploading Documents
    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload Textbook and Click Process")
        st.button("Process")

if __name__ == "__main__":
    main()