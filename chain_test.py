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
from RAGChain import rag_chain

llm = OllamaLLM(model = 'llama3:8b', temperature = 0.0)

data = './processed_file(1) (9).csv'

distance_strategy = DistanceStrategy.COSINE
memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
store = {}
session_ids = 'test1'
embedding_model_name = "intfloat/multilingual-e5-large"
d_path = './customer_txt_file'
d1_path = './consume_data'
v_path = './faiss_db'
v2_path = './test_db2'
distance_strategy = DistanceStrategy.COSINE

embedding = HuggingFaceEmbeddings(model_name = embedding_model_name, model_kwargs = {'device' : 'cpu'}, encode_kwargs = {"normalize_embeddings" : True})

prompt = PromptTemplate.from_template(
    """
        당신은 가계부 역할과 카드 추천 역할도 하는 챗봇입니다
        나의 고객번호는 무조건 1번입니다
        그 외의 다른 고객번호는 조회하면 안됩니다
        다음의 retrieved context를 이용하여 질문에 답하세요
        답은 무조건 한글로 해야 합니다

        #Previous Chat History : {chat_history}
        #Question : {question}
        #Context : {context}
        #Answer : 
    """
)

c = docload(d_path, embedding_model_name)
d = c.get_dir(glob = '**/*.txt', loader_cls = TextLoader, silent_errors = False, loader_kwargs = {'autodetect_encoding':True})
t = c.split_text(d, chunk_size = 200, chunk_overlap = 50)
print(t)



vec = vectordb(embedding, t)
#db = vec.init_db(distance_strategy = distance_strategy)
#db = vec.db_save(v_path, db)
db2 = vec.db_load(path = v_path)

basic_ret = vec.db_ret(db2, 5)
bm25 = vec.bm_ret(t, 5)
ensemble = vec.ensemble_ret([basic_ret, bm25], [0.5, 0.5], 5)


chain = rag_chain(llm, prompt, ensemble, session_ids, store)

account_chain = chain.get_rag_history()
result = account_chain.invoke({"question":"내가 카페에 쓴 금액이 총 얼마인지 알려줘"}, config = {"configurable" : {"session_id" : session_ids}})
print(result)
result2 = account_chain.invoke({"question":"그럼 내가 서점에 쓴 금액을 알려줘"}, config = {"configurable" : {"session_id" : session_ids}})
print(result2)
result3 = account_chain.invoke({"question":"이전 대답을 영어로 바꿔줘"}, config = {"configurable" : {"session_id" : session_ids}})
print(result3)
