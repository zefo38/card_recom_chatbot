from DocLoader import docload
from langchain_community.vectorstores import FAISS

class vectordb(docload):
    def __init__(self, embedding_model_name, data):
        super().__init__(self, embedding_model_name)
        self.data = data
        
    def save_embed(self):
        super().embedding()
        