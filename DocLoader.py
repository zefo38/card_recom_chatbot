import langchain
import langchain_core
import langchain_text_splitters
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
import json

path = './consumer_data.csv'
encoding = 'utf-8'
source_column = '고객번호'

class docload():
    def __init__(self):
        self.path = path
        self.encoding = encoding
        self.source_column = source_column


    def get_csv(self, path, encoding, source_column):
        loader = CSVLoader(file_path = self.path, encoding = self.encoding, source_column = self.source_column)
        data = loader.load()
        return data
    
    def get_json(self, path, schema, text_content):
        loader = JSONLoader(file_path = path, jq_schema = schema, text_content = text_content)
        j_data = loader.load()
        return j_data
    
    def json_split(max_chunk_size, js_path):
        with open(js_path, 'r') as f:
            js_data = json.load(f)
        splitter = RecursiveJsonSplitter(max_chunk_size = max_chunk_size)
        json_chunk = splitter.split_json(json_data = js_data)
        return json_chunk

c = docload()
d = c.get_csv(path, encoding, source_column)
print(d)