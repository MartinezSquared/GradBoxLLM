# GradBoxLLM

## The Textbook AI Assistant

Add course related context to your AI chat prompts! Its as simple as supplying a pdf version of your textbook!

Tired of adding a paragraph of context just so ChatGPT or Gemini understant what you're asking for? Then this is the solution for you. No longer will you have to condition your AI chat instance with relavent couse understandings. Simply upload your textbook as a pdf and have GradBoxLLM do the heavy lifting:

1. Convert the PDF to text
2. Parse the PDF text into smaller chunks
3. Embed the chunks of text as a vectors
4. Store the vectors with the corresponding text chunk in a vectorstore

The next time you ask a question, GradBoxLLM will:

5. Embed your question
6. Choose the 8 of the closest chunks of text

### Retrieval-Augmented Generation (RAG) 

AI assistant designed to help students answer textbook-related queries using FAISS for retrieval and Google Gemini AI for response generation. The app is built with Streamlit and includes user authentication.
