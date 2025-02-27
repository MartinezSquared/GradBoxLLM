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

async def main(pdf_path: str, faiss_index_path: str, query: str, 
               model: SentenceTransformer, llm: ChatGoogleGenerativeAI, 
               prompt_template: str):
    """
    Main function that processes a PDF, retrieves relevant text chunks,
    and uses an LLM for retrieval augmented generation.

    Args:
        pdf_path (str): Path to the PDF file.
        faiss_index_path (str): Path to the FAISS index on disk.
        query (str): Query to be used for text retrieval.
        model (SentenceTransformer): Pre-instantiated SentenceTransformer model.
        llm (ChatGoogleGenerativeAI): Pre-configured LLM instance.
        prompt_template (str): Template for the prompt with placeholders {context} and {query}.
    """
    # Load and split the PDF pages
    pages = await load_pdf_pages(pdf_path)
    pdf_chunks = split_pdf_pages(pages, chunk_size=200, chunk_overlap=50)
    print("Example Chunk:", pdf_chunks[300])
    
    # Compute embeddings using the provided SentenceTransformer model
    embeddings = compute_pdf_embeddings(pdf_chunks, model, device="cuda")
    
    # Create or load the FAISS index
    index = load_or_create_faiss_index(faiss_index_path, embeddings, pdf_chunks, model)
    
    # Retrieve text chunks using the provided query
    retrieved_chunks = retrieve_text_chunks(query, index, model)
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Meta Data {i+1}:")
        print(chunk.metadata)
        print(f"Chunk {i+1}:")
        print(chunk.page_content)
        print("-" * 20)
    
    # Format the prompt using the provided template
    formatted_prompt = prompt_template.format(context=retrieved_chunks[0].page_content, query=query)
    response = llm.invoke(formatted_prompt)
    print(response.content)

def compute_pdf_embeddings(docs, model: SentenceTransformer, device="cuda"):
    """
    Computes embeddings for a list of Document objects using the provided model.

    Args:
        docs (list[Document]): List of Document objects containing PDF chunks.
        model (SentenceTransformer): Pre-instantiated embedding model.
        device (str, optional): Device to use for computation. Defaults to "cuda".

    Returns:
        np.ndarray: Computed embeddings as a NumPy array.
    """
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)
    return embeddings

def split_pdf_pages(pages, chunk_size: int = 100, chunk_overlap: int = 0):
    """
    Splits the PDF pages into smaller text chunks using a recursive character splitter.

    Args:
        pages (list): List of pages from the PDF.
        chunk_size (int, optional): Desired size of each chunk. Defaults to 100.
        chunk_overlap (int, optional): Overlap between consecutive chunks. Defaults to 0.

    Returns:
        list[Document]: A list of Document objects with split text and metadata.
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

async def load_pdf_pages(file_path: str):
    """
    Asynchronously loads PDF pages from a file using PyPDFLoader.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: A list of loaded pages.
    """
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

def create_faiss_index(embeddings: np.ndarray, documents: list[Document], vector_store: Optional[FAISS] = None):
    """
    Creates or updates a FAISS index from document embeddings.

    Args:
        embeddings (np.ndarray): Array of computed embeddings.
        documents (list[Document]): List of Document objects.
        vector_store (Optional[FAISS], optional): Existing FAISS vector store. Defaults to None.

    Returns:
        FAISS: A FAISS index instance.
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
    Loads an existing FAISS index from disk if available; otherwise, creates a new one,
    saves it to disk, and returns the index.

    Args:
        faiss_index_path (str): The path to the FAISS index on disk.
        embeddings (np.ndarray): Computed embeddings for the PDF chunks.
        pdf_chunks (list[Document]): List of Document objects containing PDF chunks.
        model (SentenceTransformer): The embedding model to use for loading the index.

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
    Retrieves the top 'k' text chunks from the FAISS index that best match the query.

    Args:
        query (str): The query string.
        index (FAISS): The FAISS index.
        model (SentenceTransformer): The embedding model for query encoding.
        k (int, optional): Number of nearest neighbors to retrieve. Defaults to 8.

    Returns:
        list: A list of retrieved Document objects.
    """
    query_embedding = model.encode([query]).astype(np.float32)
    scores, indices = index.index.search(query_embedding, k)
    docs = [index.index_to_docstore_id[idx] for idx in indices[0]]
    retrieved_chunks = [index.docstore.search(doc_id) for doc_id in docs]
    return retrieved_chunks

if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    
    # Define directories and file paths
    assets_dir = "../googleDrive/assets/"
    pdf_dir = os.path.join(assets_dir, "textbooks")
    faiss_index_path = os.path.join(assets_dir, "faissIndex")
    pdf_path = os.path.join(pdf_dir, "example.pdf")
    
    # Instantiate the SentenceTransformer and ChatGoogleGenerativeAI externally
    model = SentenceTransformer("Lajavaness/bilingual-embedding-small", trust_remote_code=True, device="cuda")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp-01-21",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GOOGLE_API_KEY
    )
    
    # Define the query and prompt template with placeholders for formatting
    query = """
    An elderly client who experiences nighttime confusion wanders
    from his room into the room of another client. The nurse can best
    help decrease the client’s confusion by:
    ❍ A. Assigning a nursing assistant to sit with him until he falls asleep
    ❍ B. Allowing the client to room with another elderly client
    ❍ C. Administering a bedtime sedative
    ❍ D. Leaving a nightlight on during the evening and night shifts
    """
    prompt_template = """
    You are a helpful assistant that answers questions based on the provided context.
    Context: {context}
    Question: {query}
    Answer:
    """
    
    nest_asyncio.apply()
    asyncio.run(main(pdf_path, faiss_index_path, query, model, llm, prompt_template))


