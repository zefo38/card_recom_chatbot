from DocLoader import docload
from langchain_community.vectorstores import FAISS

class vectordb(docload):
    def __init__(self, embedding_model_name, model_kwargs, encode_kwargs):
        super().__init__(self, embedding_model_name, model_kwargs, encode_kwargs)
        
    def save_embed(self, embedding_model_name, model_kwargs, encode_kwargs):
        embed = docload.embedding(self.embedding_model_name, self.model_kwargs, self.encode_kwargs)