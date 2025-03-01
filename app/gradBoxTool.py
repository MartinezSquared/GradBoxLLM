import os
import asyncio
import nest_asyncio
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional

async def load_pdf_pages(file_path: str):
    """
    Asynchronously loads PDF pages from the given file path using PyPDFLoader.
    
    Args:
        file_path (str): The path to the PDF file.
    
    Returns:
        list: A list of pages loaded from the PDF.
    """
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

def split_pdf_pages(pages, chunk_size: int = 100, chunk_overlap: int = 0):
    """
    Splits PDF pages into smaller text chunks using RecursiveCharacterTextSplitter.
    
    Args:
        pages (list): List of pages from the PDF.
        chunk_size (int, optional): Desired size of each chunk. Defaults to 100.
        chunk_overlap (int, optional): Overlap between consecutive chunks. Defaults to 0.
    
    Returns:
        list[Document]: List of Document objects containing text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_text(page.page_content)
        for chunk in page_chunks:
            chunks.append(Document(page_content=chunk, metadata=page.metadata))
    return chunks

def compute_pdf_embeddings(docs, model: SentenceTransformer, device="cuda"):
    """
    Computes embeddings for a list of Document objects using the provided SentenceTransformer model.
    
    Args:
        docs (list[Document]): List of Document objects.
        model (SentenceTransformer): Pre-instantiated SentenceTransformer model.
        device (str, optional): Device to perform computation on. Defaults to "cuda".
    
    Returns:
        np.ndarray: Array of computed embeddings.
    """
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)
    return embeddings

def create_faiss_index(embeddings: np.ndarray, documents: list[Document], vector_store: Optional[FAISS] = None):
    """
    Creates or updates a FAISS index using the computed embeddings and associated documents.
    
    Args:
        embeddings (np.ndarray): Array of computed embeddings.
        documents (list[Document]): List of Document objects.
        vector_store (Optional[FAISS], optional): Existing FAISS vector store to update. Defaults to None.
    
    Returns:
        FAISS: A FAISS vector store instance.
    """
    embeddings = embeddings.astype("float32")
    texts = [doc.page_content for doc in documents]
    embedding_model = SentenceTransformer("Lajavaness/bilingual-embedding-small", trust_remote_code=True)
    if vector_store is None:
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=[doc.metadata for doc in documents]
        )
    else:
        vector_store.add_texts(
            texts=texts,
            embeddings=embeddings,
            metadatas=[doc.metadata for doc in documents]
        )
    return vector_store

def load_or_create_faiss_index(faiss_index_path: str, embeddings: np.ndarray, pdf_chunks: list[Document], model: SentenceTransformer) -> FAISS:
    """
    Loads an existing FAISS index from disk if available; otherwise, creates a new one.
    
    Args:
        faiss_index_path (str): Path to the FAISS index on disk.
        embeddings (np.ndarray): Computed embeddings for the PDF chunks.
        pdf_chunks (list[Document]): List of Document objects containing PDF chunks.
        model (SentenceTransformer): The embedding model.
    
    Returns:
        FAISS: The loaded or newly created FAISS index.
    """
    if os.path.exists(faiss_index_path):
        print("Loading existing FAISS index from disk...")
        index = FAISS.load_local(faiss_index_path, model, allow_dangerous_deserialization=True)
    else:
        print("Creating and saving new FAISS index...")
        index = create_faiss_index(embeddings, pdf_chunks)
        index.save_local(faiss_index_path)
    return index

def retrieve_text_chunks(query: str, index: FAISS, model: SentenceTransformer, k: int = 8):
    """
    Retrieves the top k text chunks from the FAISS index that best match the query.
    
    Args:
        query (str): The query string.
        index (FAISS): The FAISS index.
        model (SentenceTransformer): The embedding model.
        k (int, optional): Number of nearest neighbors to retrieve. Defaults to 8.
    
    Returns:
        list: List of retrieved Document objects.
    """
    query_embedding = model.encode([query]).astype(np.float32)
    scores, indices = index.index.search(query_embedding, k)
    docs = [index.index_to_docstore_id[idx] for idx in indices[0]]
    retrieved_chunks = [index.docstore.search(doc_id) for doc_id in docs]
    return retrieved_chunks

def create_rag_chain_with_memory(pdf_path: str, 
                                 faiss_index_path: str, 
                                 model: SentenceTransformer, 
                                 llm: ChatGoogleGenerativeAI, 
                                 prompt_template: str):
    """
    Creates a Retrieval Augmented Generation (RAG) chain with conversation memory.
    
    This function processes a PDF, computes embeddings, and sets up a FAISS index. It returns
    a callable function that accepts a query, retrieves relevant text, incorporates prior conversation
    history into the prompt, and returns the LLM’s response. The conversation history is maintained
    across calls.
    
    Args:
        pdf_path (str): Path to the PDF file.
        faiss_index_path (str): Path to the FAISS index on disk.
        model (SentenceTransformer): Pre-instantiated SentenceTransformer model.
        llm (ChatGoogleGenerativeAI): Pre-configured LLM instance.
        prompt_template (str): A template containing placeholders {chat_history}, {context}, and {query}.
    
    Returns:
        function: A callable that takes a query string and returns the LLM's response.
    """
    # Load and process PDF pages.
    pages = asyncio.run(load_pdf_pages(pdf_path))
    pdf_chunks = split_pdf_pages(pages, chunk_size=200, chunk_overlap=50)
    embeddings = compute_pdf_embeddings(pdf_chunks, model, device="cuda")
    index = load_or_create_faiss_index(faiss_index_path, embeddings, pdf_chunks, model)
    
    # Initialize conversation history as a list of messages.
    conversation_history = []

    def rag_chain(query: str) -> str:
        """
        Executes the RAG chain for a given query while maintaining conversation history.
        
        Args:
            query (str): The user query.
        
        Returns:
            str: The LLM's response.
        """
        # Append the new user query to the history.
        conversation_history.append(f"User: {query}")
        
        # Retrieve relevant text chunk.
        retrieved_chunks = retrieve_text_chunks(query, index, model)
        # Assume the first chunk is most relevant.
        context_text = retrieved_chunks[0].page_content if retrieved_chunks else ""
        
        # Combine conversation history into a single string.
        history_str = "\n".join(conversation_history)
        
        # Format the prompt to include conversation history, context, and current query.
        formatted_prompt = prompt_template.format(
            chat_history=history_str, 
            context=context_text, 
            query=query
        )
        
        # Invoke the LLM with the combined prompt.
        response = llm.invoke(formatted_prompt)
        
        # Append the assistant's response to the conversation history.
        conversation_history.append(f"Assistant: {response.content}")
        return response.content

    return rag_chain

if __name__ == "__main__":
    # Load environment variables from .env file.
    load_dotenv()
    
    # Retrieve API keys and define file paths.
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    assets_dir = "../googleDrive/assets/"
    pdf_dir = os.path.join(assets_dir, "textbooks")
    faiss_index_path = os.path.join(assets_dir, "faissIndex")
    pdf_path = os.path.join(pdf_dir, "example.pdf")
    
    # Instantiate the SentenceTransformer and LLM.
    model = SentenceTransformer("Lajavaness/bilingual-embedding-small", trust_remote_code=True, device="cuda")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp-01-21",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GOOGLE_API_KEY
    )
    
    # Define a prompt template with placeholders for conversation history, context, and query.
    prompt_template = (
        "You are a helpful assistant.\n"
        "Previous conversation:\n{chat_history}\n"
        "Context from document: {context}\n"
        "User Query: {query}\n"
        "Answer:"
    )
    
    # Apply nest_asyncio to allow asyncio.run in interactive environments.
    nest_asyncio.apply()
    
    # Create the RAG chain with conversation memory.
    rag_chain = create_rag_chain_with_memory(pdf_path, faiss_index_path, model, llm, prompt_template)
    
    # Example query.
    query = (
        "An elderly client who experiences nighttime confusion wanders "
        "from his room into the room of another client. The nurse can best "
        "help decrease the client’s confusion by:\n"
        "❍ A. Assigning a nursing assistant to sit with him until he falls asleep\n"
        "❍ B. Allowing the client to room with another elderly client\n"
        "❍ C. Administering a bedtime sedative\n"
        "❍ D. Leaving a nightlight on during the evening and night shifts"
    )
    
    # Execute the chain and print the response.
    response = rag_chain(query)
    print("Final Response:")
    print(response)

