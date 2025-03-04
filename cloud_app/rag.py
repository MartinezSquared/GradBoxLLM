import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

def load_vectorstore(faiss_path: str = "./faissIndex"):
    """
    Loads the FAISS vectorstore from the specified path using a CPU-based SentenceTransformer.
    
    :param faiss_path: The directory or file path containing the FAISS index.
    :return: A FAISS index if found; otherwise, None.
    """
    embed_model = SentenceTransformer(
        "Lajavaness/bilingual-embedding-small",
        trust_remote_code=True,
        device="cpu"
    )
    if os.path.exists(faiss_path):
        index = FAISS.load_local(
            faiss_path,
            embed_model,
            allow_dangerous_deserialization=True
        )
        return index
    else:
        return None

def retrieve_text_chunks(query: str, index: FAISS, k: int = 4):
    """
    Retrieves the top k most similar text chunks to a query from the FAISS index.
    """
    model = SentenceTransformer(
        "Lajavaness/bilingual-embedding-small",
        trust_remote_code=True,
        device="cpu"
    )
    query_embedding = model.encode([query]).astype(np.float32)
    scores, indices = index.index.search(query_embedding, k)
    doc_ids = [index.index_to_docstore_id[idx] for idx in indices[0]]
    retrieved_chunks = []
    for doc_id in doc_ids:
        retrieved_chunks.append(index.docstore.search(doc_id))
    return retrieved_chunks

def create_faiss_index(documents: list[Document]):
    """Creates a FAISS index from a list of documents."""
    model = SentenceTransformer(
        "Lajavaness/bilingual-embedding-small",
        trust_remote_code=True,
        device="cpu"
    )
    embeddings = model.encode([doc.page_content for doc in documents])
    embeddings = embeddings.astype("float32")
    faiss_index = FAISS.from_embeddings(
        text_embeddings=list(
            zip([doc.page_content for doc in documents], embeddings)
        ),
        embedding=model,
        metadatas=[doc.metadata for doc in documents]
    )
    return faiss_index

def save_faiss_index(index: FAISS, path: str = "./faissIndex"):
    """Saves a FAISS index to the specified path."""
    os.makedirs(path, exist_ok=True)
    index.save_local(path)
    print(f"Vectorstore saved to: {path}")

if __name__ == "__main__":
    # Example usage: Load the FAISS index and run a quick retrieval
    # Load environment variables from .env file (including GOOGLE_API_KEY)
    load_dotenv()

    # Constants
    FAISS_INDEX_PATH = "./faissIndex"

    index = load_vectorstore(FAISS_INDEX_PATH)
    if index is not None:
        print("FAISS index loaded successfully.")
        
        # Sample query retrieval
        query = "Explain how to prepare medication for a patient."
        retrieved_chunks = retrieve_text_chunks(query, index, k=2)
        print(f"Top 2 retrieved chunks for query '{query}':")
        for chunk in retrieved_chunks:
            print("----")
            print("Content:", chunk.page_content)
            print("Metadata:", chunk.metadata)
    else:
        print(f"No FAISS index found at '{FAISS_INDEX_PATH}'. Please build and save an index first.")

