from langchain.memory import VectorStoreRetrieverMemory
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from RAG_VectorDB import vectordb


class ConvMemory():
    