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
    
    def db_save(self, path, vectorstore):
        vectorstore = vectorstore
        return vectorstore.save_local(path)
    
    def db_load(self, path):
        loaded_db = FAISS.load_local(path, self.embedding, allow_dangerous_deserialization=True)
        return loaded_db
    
    def ensemble_ret(self, rets, weights, c):
        ret = EnsembleRetriever(retrievers = rets, weights = weights, c = c)
        return ret