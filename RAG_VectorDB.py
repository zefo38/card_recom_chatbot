from DocLoader import docload
from langchain_community.vectorstores import FAISS

faiss_db = FAISS.from_documents()

class vectordb(docload):
    def __init__(self, embedding_model_name, model_kwargs, encode_kwargs, input_data, db, query):
        super().__init__(self, embedding_model_name)
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.input_data = input_data
        self.db = db
        self.query = query
        
    def save_embed(self, db):
        embed = docload.embedding(self.embedding_model_name, self.model_kwargs, self.encode_kwargs, self.input_data)
        db = self.db(self.input_data, embed)
        return db
    
    def querying(self, query):
        