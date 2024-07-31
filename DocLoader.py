import langchain
import langchain_core
import langchain_text_splitters
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
import json
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

embedding_model_name = "intfloat/multilingual-e5-large"

path = './consumer_data.csv'
encoding = 'utf-8'
source_column = '고객번호'
d_path = './source2'

class docload():
    def __init__(self, path, d_path, embedding_model_name):
        self.path = path
        self.d_path = os.path.abspath(d_path)
        self.embedding_model_name = embedding_model_name

    def get_csv(self, encoding, source_column):
        loader = CSVLoader(self.path, encoding = encoding, source_column = source_column)
        data = loader.load()
        return data
    
    def get_dir(self, glob, loader_cls, silent_errors, loader_kwargs):
        loader = DirectoryLoader(self.d_path, glob = glob, loader_cls = loader_cls, silent_errors = silent_errors, loader_kwargs = loader_kwargs)
        d_data = loader.load_and_split()
        return d_data

    
    def embedding(self, model_kwargs, encode_kwargs, chunked_data):
        hf_embedding = HuggingFaceEmbeddings(model_name = self.embedding_model_name)
        get_embed = hf_embedding.embed_documents(chunked_data, **model_kwargs, **encode_kwargs)
        return get_embed


c = docload(path, d_path, embedding_model_name)
d = c.get_dir(glob = '**/*.tsv', loader_cls = TextLoader, silent_errors = False, loader_kwargs = {'autodetect_encoding':True})
embed = c.embedding(model_kwargs = {"device" : "cpu"}, encode_kwargs = {"normalize_embeddings" : True}, chunked_data = d)
print(embed)