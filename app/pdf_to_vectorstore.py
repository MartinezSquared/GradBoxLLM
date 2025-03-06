import os
import asyncio
import nest_asyncio
from langchain_community.document_loaders import pypdfloader
from langchain_text_splitters import recursivecharactertextsplitter
from langchain_core.documents import document
from sentence_transformers import sentencetransformer
import numpy as np
from langchain_community.vectorstores import faiss
# from langchain_community.embeddings import huggingfaceembeddings
from langchain_huggingface import huggingfaceembeddings

# allow nested asyncio loops (useful in notebooks or certain runtime environments)
nest_asyncio.apply()

async def load_pdf_pages(file_path: str):
    """asynchronously loads pdf pages from the given file path."""
    loader = pypdfloader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

def split_pdf_pages(pages, chunk_size: int = 100, chunk_overlap: int = 0):
    """
    splits pdf pages into smaller text chunks, preserving metadata.
    """
    text_splitter = recursivecharactertextsplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_text(page.page_content)
        for chunk in page_chunks:
            chunks.append(document(page_content=chunk, metadata=page.metadata))
    return chunks

def compute_pdf_embeddings(docs, device: str = "cuda"):
    """
    computes embeddings for a list of document objects using a sentencetransformer model.
    """
    texts = [doc.page_content for doc in docs]
    model = sentencetransformer(
        "lajavaness/bilingual-embedding-small",
        trust_remote_code=true,
        device=device
    )
    embeddings = model.encode(texts)
    return embeddings

def create_faiss_index(embeddings: np.ndarray, documents: list[document], vector_store: 'optional[faiss]' = none) -> faiss:
    """
    creates or updates a faiss index (vector store) based on provided embeddings and documents.
    """
    embeddings = embeddings.astype("float32")
    texts = [doc.page_content for doc in documents]
    embedding_model = huggingfaceembeddings(
        model_name="lajavaness/bilingual-embedding-small",
        model_kwargs={'trust_remote_code': true}
    )
    if vector_store is none:
        vector_store = faiss.from_texts(
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

import os

def get_pdf_filenames(folder_path):
    pdf_files = []
    try:
        #ensure folder_path is an absolute path.
        absolute_folder_path = os.path.abspath(folder_path)
        for filename in os.listdir(absolute_folder_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(absolute_folder_path, filename)
                pdf_files.append(full_path)
    except filenotfounderror:
        print(f"error: folder '{folder_path}' not found.")
    except notadirectoryerror:
        print(f"error: '{folder_path}' is not a directory.")
    except exception as e:
        print(f"an unexpected error occurred: {e}")
    print(pdf_files)
    return pdf_files

if __name__ == "__main__":
    # list your pdf file paths (adjust the paths as needed)
    folder_path = "../googledrive/assets/textbooks/"  # adjust as needed
    pdf_paths = get_pdf_filenames(folder_path)
    # pdf_paths = [
    #     "../googledrive/assets/textbooks/medicalsurgicalnursing.pdf",
    #     "../googledrive/assets/textbooks/medical-surgical_nursing--preparation_for_practice_(2010).pdf"
    # ]
    
    all_chunks = []
    # process each pdf file iteratively
    for pdf_path in pdf_paths:
        print(f"processing: {pdf_path}")
        pages = asyncio.run(load_pdf_pages(pdf_path))
        pdf_chunks = split_pdf_pages(pages, chunk_size=200, chunk_overlap=50)
        all_chunks.extend(pdf_chunks)
    
    print(f"total number of chunks: {len(all_chunks)}")
    
    # compute embeddings (use "cuda" if you have a compatible gpu, otherwise "cpu")
    embeddings = compute_pdf_embeddings(all_chunks, device="cuda")
    
    # define a local path for saving the faiss index (it will be a directory)
    faiss_index_path = "./faissindex"
    index = create_faiss_index(embeddings, all_chunks)
    
    # create directory if it does not exist
    os.makedirs(faiss_index_path, exist_ok=true)
    index.save_local(faiss_index_path)
    print(f"vectorstore saved to: {faiss_index_path}")

