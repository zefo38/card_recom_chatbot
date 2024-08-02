from DocLoader import docload
from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever


class vectordb():
    def __init__(self, embedding, input_data):
        self.embedding = embedding
        self.input_data = input_data

        
    def init_db(self, distance_strategy):
        db = FAISS.from_documents(self.input_data, self.embedding, distance_strategy = distance_strategy)
        return db
    
    def db_save(self, index_name, path):
        return FAISS.save_local(folder_path=path, index_name = index_name)
    
    def db_load(self, path, index_name):
        loaded_db = FAISS.load_local(path, self.embedding, index_name = index_name)
        return loaded_db
    
    def ensemble_ret(self, rets, weights, c):
        ret = EnsembleRetriever(retrievers = rets, weights = weights, c = c)
        return ret