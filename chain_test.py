from DocLoader import docload
from RAG_VectorDB import vectordb
import langchain
from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.memory import ConversationBufferMemory
import faiss
from langchain_community.docstore import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoModelForCausalLM
from langchain_ollama.llms import OllamaLLM
from transformers import AutoModel
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import pandas as pd
from RAGChain import rag_chain

llm = OllamaLLM(model = 'llama3:8b', temperature = 0.0)

data = './processed_file(1) (9).csv'

distance_strategy = DistanceStrategy.COSINE
store = {}
session_ids = 'test2'
embedding_model_name = 'intfloat/multilingual-e5-large'
d_path = './customer_id_1'
v_path = './Faiss_DB'
v_path2 = './pdf_db'
distance_strategy = DistanceStrategy.COSINE

embedding = HuggingFaceEmbeddings(model_name = embedding_model_name, model_kwargs = {'device' : 'cpu'}, encode_kwargs = {"normalize_embeddings" : True})



c = docload(d_path, embedding_model_name)
d = c.get_dir(glob = '**/*.txt', loader_cls = TextLoader, silent_errors = False, loader_kwargs = {'autodetect_encoding':True})
pdf = c.pdf_dir(glob = '**/*.pdf', silent_errors = False)
t = c.split_text(d, chunk_size = 50, chunk_overlap = 0)
t2 = c.split_text(pdf, chunk_size = 50, chunk_overlap = 0)



vec = vectordb(embedding, t)
vec2 = vectordb(embedding, t2)
db = vec.init_db(distance_strategy = distance_strategy)
db2 = vec2.init_db(distance_strategy=distance_strategy)
db_merge = vec.merge_db(db, db2)
db_merge = vec.db_save(v_path, db_merge)
db_loaded = vec.db_load(path = v_path)

basic_ret = vec.db_ret(db_loaded, 10)
bm25 = vec.bm_ret(t, 10)
ensemble = vec.ensemble_ret([basic_ret, bm25], [0.8, 0.2], 10)




chain = rag_chain(llm, ensemble, session_ids, store)

account_chain = chain.get_rag_history()
result = account_chain.invoke({"question":"내가 카페에서 얼마나 썼어?"}, config = {"configurable" : {"session_id" : session_ids}})
print(result)
result2 = account_chain.invoke({"question":"내가 서점에서 쓴 금액을 알려줘"}, config = {"configurable" : {"session_id" : session_ids}})
print(result2)
result3 = account_chain.invoke({"question":"그러면 서점에 많이썼어? 아니면 카페에 많이썼어?"}, config = {"configurable" : {"session_id" : session_ids}})
print(result3)
result4 = account_chain.invoke({"question" : "그 둘을 비교해서 나에게 맞는 카드를 추천해줘"}, config = {"configurable" : {"session_id" : session_ids}})
print(result4)
result5 = account_chain.invoke({"question" : "나에게 혜택이 좋은 카드를 추천해줘"}, config = {"configurable" : {"session_id" : session_ids}})
print(result5)
result6 = account_chain.invoke({"question" : "나에게 혜택이 좋은 카드를 세개 추천해줘"}, config = {"configurable" : {"session_id" : session_ids}})
print(result6)

