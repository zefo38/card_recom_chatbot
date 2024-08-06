from langchain_model import chat_chain
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

llm = OllamaLLM(model = 'llama3:8b', temperature = 0.0)

data = './processed_file(1) (9).csv'

distance_strategy = DistanceStrategy.COSINE
memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
store = {}
session_ids = 'test1'
embedding_model_name = "intfloat/multilingual-e5-large"
d_path = './source4'
v_path = './test_db'
distance_strategy = DistanceStrategy.COSINE

embedding = HuggingFaceEmbeddings(model_name = embedding_model_name, model_kwargs = {'device' : 'cpu'}, encode_kwargs = {"normalize_embeddings" : True})



c = docload(d_path, embedding_model_name)
d = c.get_dir(glob = '**/*.tsv', loader_cls = CSVLoader, silent_errors = False, loader_kwargs = {'autodetect_encoding':True})
t = c.split_text(d, chunk_size = 200, chunk_overlap = 50)



vec = vectordb(embedding, d)
db = vec.init_db(distance_strategy=distance_strategy)
db = vec.db_save(v_path, db)
db2 = vec.db_load(path = v_path)

basic_ret = vec.db_ret(db2, 2)
bm25 = vec.bm_ret(d, 2)
ensemble = vec.ensemble_ret([basic_ret, bm25], [0.5, 0.5], 2)



chain = chat_chain(llm, memory, ensemble)

account_chain = chain.get_chain_account(data_path = data)
result = account_chain.predict(query = "내가 카페에 총 얼마 썼는지 알려줘")
print(result)
context = chain.save_memory("내가 카페에 총 얼마 썼는지 알려줘", result)
result2 = account_chain.predict(query = "그럼 서점엔 얼마나 썼는지 알려줘")
print(result2)
context = chain.save_memory("그럼 서점엔 얼마나 썼는지 알려줘", result2)
result3 = account_chain.predict(query = "그 둘의 총합을 알려줘")
context = chain.save_memory("그 둘의 총합을 알려줘", result3)
print(result3)
