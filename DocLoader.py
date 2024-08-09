import langchain
import langchain_core
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
import json
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from multiprocess import Pool
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader


embedding_model_name = "intfloat/multilingual-e5-large"

path = './consumer_data.csv'
encoding = 'utf-8'
source_column = '고객번호'
d_path = './source2'

class docload():
    def __init__(self, d_path, embedding_model_name):
        self.d_path = os.path.abspath(d_path)
        self.embedding_model_name = embedding_model_name

    
    def get_dir(self, glob, loader_cls, silent_errors, loader_kwargs):
        loader = DirectoryLoader(self.d_path, glob = glob, loader_cls = loader_cls, silent_errors = silent_errors, loader_kwargs = loader_kwargs)
        d_data = loader.load()
        return d_data
    
    def pdf_dir(self, glob, silent_errors):
        loader = PyPDFDirectoryLoader(self.d_path, glob = glob, silent_errors = silent_errors)
        pdf_data = loader.load()
        return pdf_data

    def split_text(self, t_data, chunk_size, chunk_overlap):
        spliter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        if isinstance(t_data, str) == False:
            text = spliter.split_documents(t_data)
        else:
            text = spliter.split_text(t_data)
        return text

    
    def embedding(self, model_kwargs, encode_kwargs, chunked_data):
        hf_embedding = HuggingFaceEmbeddings(model_name = self.embedding_model_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)
        get_embed = hf_embedding.embed_documents(chunked_data)
        return get_embed


