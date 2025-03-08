import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
import random

# Load environment variables from .env file (including GOOGLE_API_KEY)
load_dotenv()

# Define the FAISS index path (must match the one used in build_vectorstore.py)
faiss_index_path = "./faissIndex"

# Load the embedding model (must be the same as used when creating the index)
model = SentenceTransformer(
    "Lajavaness/bilingual-embedding-small",
    trust_remote_code=True,
    device="cuda"  # Change to "cuda" if available
)

# Load the FAISS index from disk
index = FAISS.load_local(faiss_index_path, model, allow_dangerous_deserialization=True)
print("FAISS index loaded.")

def get_faiss_index_path(base_path="./tmp/faissIndex"):
    """
    Checks if the default FAISS index path exists. If it does, generates a unique path.
    """
    faiss_index_path = base_path

    # If FAISS index already exists, generate a unique one
    while os.path.exists(faiss_index_path):
        random_int = random.randint(10000000000000000, 99999999999999999)
        faiss_index_path = f"./tmp/faissIndex{random_int}"

    return faiss_index_path

def retrieve_text_chunks(query: str, index: FAISS, k: int = 4):
    """
    Retrieves the top k most similar text chunks to a query from the FAISS index.
    """
    query_embedding = model.encode([query]).astype(np.float32)
    scores, indices = index.index.search(query_embedding, k)
    doc_ids = [index.index_to_docstore_id[idx] for idx in indices[0]]
    retrieved_chunks = []
    for doc_id in doc_ids:
        retrieved_chunks.append(index.docstore.search(doc_id))
    return retrieved_chunks


def load_vectorstore():
    """
    Loads the FAISS vectorstore from disk using a CPU-based SentenceTransformer.
    """
    embed_model = SentenceTransformer(
        "Lajavaness/bilingual-embedding-small",
        trust_remote_code=True,
        device="cuda"
    )
    if os.path.exists(FAISS_INDEX_PATH):
        from langchain_community.vectorstores import FAISS  # local import for clarity
        index = FAISS.load_local(FAISS_INDEX_PATH, embed_model, allow_dangerous_deserialization=True)
        return index
    else:
        return None


def build_vectorstore(pdf_files):
    """Process uploaded PDFs, build a vectorstore, and save it locally."""
    all_chunks = []
    for pdf_file in pdf_files:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        st.write(f"Processing: {tmp_path}")
        pages = asyncio.run(load_pdf_pages(tmp_path))
        chunks = split_pdf_pages(pages, chunk_size=200, chunk_overlap=50)
        all_chunks.extend(chunks)
        os.remove(tmp_path)
    st.write(f"Total number of chunks: {len(all_chunks)}")
    # Compute embeddings (use "cuda" if available; otherwise "cpu")
    embeddings = compute_pdf_embeddings(all_chunks, device="cuda")

    # Create the FAISS vectorstore
    index = create_faiss_index(embeddings, all_chunks)

    # Save the index locally
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    index.save_local(FAISS_INDEX_PATH)
    return index


if __name__ == "__main__":
    # Retrieve the Gemini API key from environment variables
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if GOOGLE_API_KEY is None:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp-01-21",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GOOGLE_API_KEY
    )
    
    # Define your example query
    query = """
    An elderly client who experiences nighttime confusion wanders
    from his room into the room of another client. The nurse can best
    help decrease the client’s confusion by:
    ❍ A. Assigning a nursing assistant to sit with him until he falls asleep
    ❍ B. Allowing the client to room with another elderly client
    ❍ C. Administering a bedtime sedative
    ❍ D. Leaving a nightlight on during the evening and night shifts
    """
    
    # Retrieve similar document chunks from the FAISS index
    retrieved_chunks = retrieve_text_chunks(query, index, k=4)
    retrieved_text = ""
    for chunk in retrieved_chunks:
        retrieved_text += chunk.page_content + "\n"
    
    # Compose the prompt for Gemini
    prompt = f"""
System:
You are a helpful nursing assistant that uses relevant context from a textbook and your advance reasoning to help answer the user's question."
Context:
{retrieved_text}

Question:
{query}

Answer:
    """
    
    # Only print the input prompt and the output from Gemini.
    print("Input to Gemini:")
    print(prompt)
    
    response = llm.invoke(prompt)
    
    print("\nOutput from Gemini:")
    print(response.content)
