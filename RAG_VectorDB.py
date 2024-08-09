from DocLoader import docload
from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community import vectorstores


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
    
    def db_ret(self, db, k):
        db_retriever = db.as_retriever(search_type = 'mmr', search_kwargs = {"k" : k})
        return db_retriever
    
    def bm_ret(self, data, k):
        bmret = BM25Retriever.from_documents(data, k = k)
        return bmret
    
    def ensemble_ret(self, rets, weights, c):
        ret = EnsembleRetriever(retrievers = rets, weights = weights, c = c)
        return ret
    
    def merge_db(self, db1, db2):
        db1.merge_from(db2)
        return db1