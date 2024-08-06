from DocLoader import docload
from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from multiprocess import Pool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from RAG_VectorDB import vectordb
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents.base import Document
import numpy as np
import pickle


embedding_model_name = "intfloat/multilingual-e5-large"

path = './consumer_data.csv'
encoding = 'utf-8'
source_column = '고객번호'
d_path = './source3'
v_path = './card_chatbot'
distance_strategy = DistanceStrategy.COSINE
index_name = 'test1'

embedding = HuggingFaceEmbeddings(model_name = embedding_model_name, model_kwargs = {'device' : 'cpu'}, encode_kwargs = {"normalize_embeddings" : True})

c = docload(d_path, embedding_model_name)
d = c.get_dir(glob = '**/*.tsv', loader_cls = CSVLoader, silent_errors = False, loader_kwargs = {'autodetect_encoding':True})
t = c.split_text(d, chunk_size = 200, chunk_overlap = 50)



vec = vectordb(embedding, d)
db = vec.init_db(distance_strategy=distance_strategy)
db = vec.db_save(v_path, db)
db2 = vec.db_load(path = v_path)
print(db2)
