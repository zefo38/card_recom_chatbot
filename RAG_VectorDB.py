from DocLoader import docload
from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever


class vectordb():
    def __init__(self, embedding_model_name, model_kwargs, encode_kwargs, input_data):
        self.embedding_model_name = embedding_model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.input_data = input_data

        
    def init_db(self, embedding, distance_strategy):
        db = FAISS.from_documents(self.input_data, embedding = embedding, distance_strategy = distance_strategy)
        return db
    
    def db_save(self, db, path):
        return db.save_local(folder_path=path)
    
    def db_load(self, path, embedding):
        loaded_db = FAISS.load_local(path, embedding)
        return loaded_db
    
    def ensemble_ret(self, rets, weights, c):
        ret = EnsembleRetriever(retrievers = rets, weights = weights, c = c)
        return ret