import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# Load environment variables from .env file (including GOOGLE_API_KEY)
load_dotenv()

# Constants
FAISS_INDEX_PATH = "./faissIndex"

def load_vectorstore():
    """
    Loads the FAISS vectorstore from disk using a CPU-based SentenceTransformer.
    """
    embed_model = SentenceTransformer(
        "Lajavaness/bilingual-embedding-small",
        trust_remote_code=True,
        device="cpu"
    )
    if os.path.exists(FAISS_INDEX_PATH):
        index = FAISS.load_local(FAISS_INDEX_PATH, embed_model, allow_dangerous_deserialization=True)
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
        text_embeddings=list(zip([doc.page_content for doc in documents],embeddings)),
        embedding=model,
        metadatas=[doc.metadata for doc in documents]
    )
    return faiss_index

def save_faiss_index(index: FAISS, path: str = FAISS_INDEX_PATH):
    """Saves a FAISS index to the specified path."""
    os.makedirs(path, exist_ok=True)
    index.save_local(path)
    print(f"Vectorstore saved to: {path}")
