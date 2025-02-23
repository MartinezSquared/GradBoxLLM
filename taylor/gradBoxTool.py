# Dependancies
# pip install -qU langchain_community
# pip install -qU langchain-text-splitters
# pip install -qU nest_asyncio
# pip install -qU pypdf
# pip install -qU tiktoken
# pip install -qU langchain-huggingface
# pip install -qU sentence-transformers
# pip install -qU faiss-cpu
# pip install -qU faiss-gpu
# pip install -qU langchain-google-genai

from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from langchain_core.documents import Document


