import langchain
import langchain_core
import langchain_text_splitters
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
import json
from dotenv import load_dotenv
load_dotenv()
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

model_name = "intfloat/multilingual-e5-large-instruct"

path = './consumer_data.csv'
encoding = 'utf-8'
source_column = '고객번호'
j_path = './045/01-1/Training/labeled'

class docload():
    def __init__(self, path, j_path, embedding_model_name):
        self.path = path
        self.j_path = j_path
        self.embedding_model_name = embedding_model_name

    def get_csv(self, path, encoding, source_column):
        loader = CSVLoader(file_path = self.path, encoding = encoding, source_column = source_column)
        data = loader.load()
        return data
    
    def get_json(self, j_path, schema, text_content, max_chunk_size):
        loader = JSONLoader(file_path = self.j_path, jq_schema = schema, text_content = text_content)
        j_data = loader.load()
        splitter = RecursiveJsonSplitter(max_chunk_size = max_chunk_size)
        json_chunk = splitter.split_json(json_data = j_data)
        return json_chunk
    
    def embedding(self, embedding_model_name, model_kwarg, encode_k, chunked_data):
        hf_embedding = HuggingFaceEmbeddings(model_name = self.embedding_model_name, model_kwargs = model_kwarg, encode_kwargs = encode_k)
        get_embed = hf_embedding.embed_documents(chunked_data)
        return get_embed

c = docload(path, j_path, model_name)
d = c.get_json(, )
print(d)