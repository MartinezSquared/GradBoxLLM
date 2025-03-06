import os
import asyncio
import nest_asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional

# Allow nested asyncio loops
nest_asyncio.apply()

async def load_pdf_pages(file_path: str):
    """Asynchronously loads PDF pages from the given file path."""
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

def split_pdf_pages(pages, chunk_size: int = 100, chunk_overlap: int = 0):
    """Splits PDF pages into smaller text chunks, preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_text(page.page_content)
        for chunk in page_chunks:
            chunks.append(Document(page_content=chunk, metadata=page.metadata))
    return chunks

def compute_pdf_embeddings(docs, device: str = "cuda"):
    """Computes embeddings for a list of Document objects using a SentenceTransformer model."""
    texts = [doc.page_content for doc in docs]
    model = SentenceTransformer(
        "lajavaness/bilingual-embedding-small",
        trust_remote_code=True,
        device=device
    )
    embeddings = model.encode(texts)
    return embeddings

def create_faiss_index(embeddings: np.ndarray, documents: list[Document], vector_store: Optional[FAISS] = None) -> FAISS:
    """Creates or updates a FAISS index (vector store) based on provided embeddings and documents."""
    embeddings = embeddings.astype("float32")
    texts = [doc.page_content for doc in documents]
    embedding_model = HuggingFaceEmbeddings(
        model_name="lajavaness/bilingual-embedding-small",
        model_kwargs={'trust_remote_code': True}
    )
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

def get_pdf_filenames(folder_path):
    pdf_files = []
    try:
        absolute_folder_path = os.path.abspath(folder_path)
        for filename in os.listdir(absolute_folder_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(absolute_folder_path, filename)
                pdf_files.append(full_path)
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except NotADirectoryError:
        print(f"Error: '{folder_path}' is not a directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    print(pdf_files)
    return pdf_files

if __name__ == "__main__":
    folder_path = "../googledrive/assets/textbooks/"
    pdf_paths = get_pdf_filenames(folder_path)

    all_chunks = []
    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path}")
        pages = asyncio.run(load_pdf_pages(pdf_path))
        pdf_chunks = split_pdf_pages(pages, chunk_size=200, chunk_overlap=50)
        all_chunks.extend(pdf_chunks)

    print(f"Total number of chunks: {len(all_chunks)}")

    embeddings = compute_pdf_embeddings(all_chunks, device="cuda")

    faiss_index_path = "./faissindex"
    index = create_faiss_index(embeddings, all_chunks)

    os.makedirs(faiss_index_path, exist_ok=True)
    FAISS.save_local(index, faiss_index_path)
    print(f"Vectorstore saved to: {faiss_index_path}")

