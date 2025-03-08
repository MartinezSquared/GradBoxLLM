# GradBoxLLM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./public_app/gradapp.py)

## Achieve Textbook-Level Context Using RAG

Add a textbook to your AI chat prompts! Its as simple as uploading a pdf version of your textbook.

Tired of manually adding context to ChatGPT just so they can understand you? Then this is the solution for you. No longer will you have to condition your questions with relavent information manually. Simply upload your textbook as a pdf and have GradBoxLLM do the heavy lifting.

## What is RAG

RAG is Retrieval-Augmented Generation, and it is a tool to find text that is semantically related to your question. 

When you upload a pdf to GradBoxLLM, it goes through a series of preprocessing steps to make them searchable by similarity. Think of it as a better Ctrl-f that searches based on the context of the text instead of exact matches.

RAG isn't like training and finetuning, it's much simpler, and it doesn't rely on a large amount of computing power. 
RAG is used to engineer your prompt with relavent information. To prepare for RAG, the PDF text is stored **with** a vector.
The vector's direction and magnitude is a mapping of its semantic meaning.
This is useful when prompting an LLM because your question will be accompanied with relavent text chunks. 
This is a accomplished by pairing your question with a vector to find text chunks with similar vector directions and magnitues.

## What Happens When I Upload My PDF?

### PDF to Vectorstore 

1. Convert the PDF to text using OCR
2. Parse the text into smaller chunks
3. Embed the text chunks as vectors
4. Store and save the text chunks with the vectors in a vectorstore

### Retrieval-Augmented Generation to Gemini LLM

5. Load the vectorstore
6. Ask a question
7. Embed the question as a vector
8. Search the vectorstore for 8 vectors that are most similar to the question vector
9. Retrieve the text chunks corresponding to the 8 most similar vectors
10. Supply the text chunks and question to the Gemini Bot
11. Gemini LLM will use the text chunks to answer the question
12. The user will be able to see what was supplied to Gemini and check the sources

## How to Use GradBox 

