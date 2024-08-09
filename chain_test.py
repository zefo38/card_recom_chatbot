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
store = {}
session_ids = 'test2'
embedding_model_name = 'intfloat/multilingual-e5-large'
d_path = './customer_txt_file'
v_path = './faiss_db'
distance_strategy = DistanceStrategy.COSINE

embedding = HuggingFaceEmbeddings(model_name = embedding_model_name, model_kwargs = {'device' : 'cpu'}, encode_kwargs = {"normalize_embeddings" : True})

prompt = PromptTemplate.from_template(
    """
        당신은 가계부 역할을 하는 챗봇입니다.

        - 챗봇 사용자의 고객번호는 무조건 1번입니다.
        - 고객번호 1번 외의 다른 고객번호는 무시하세요.
        - 질문에 특정 카테고리만 언급되었을 경우, 해당 카테고리에서 모든 날짜에 고객번호 1번이 쓴 금액 총합을 알려주세요.
            - 예: "내가 카페에서 쓴 금액은 얼마인가요?"라고 물어보면, 고객번호 1번이 모든 날짜에 카페에서 쓴 총 금액을 답변하세요.
        - 질문에 특정 날짜만 언급되었을 경우, 해당 날짜에 모든 카테고리에서 고객번호 1번이 쓴 금액 총합을 알려주세요.
            -  예: "내가 2023년 5월 1일에 쓴 금액은 얼마인가요?"라고 물어보면, 고객번호 1번이 2023년 5월 1일에 모든 카테고리에서 쓴 총 금액을 답변하세요.
        - 질문에 카테고리와 날짜가 동시에 언급되었을 경우, 해당 날짜에 해당 카테고리에서 고객번호 1번이 쓴 금액 총합을 알려주세요.
            - 예: "내가 2023년 5월 1일에 카페에서 쓴 금액은 얼마인가요?"라고 물어보면, 고객번호 1번이 2023년 5월 1일에 카페에서 쓴 총 금액을 답변하세요.
        - 내가 비교나 계산을 해달라고 요청하면 0원이더라도 무조건 비교나 계산을 해줘.
        
        - 답은 무조건 한글로 해야 합니다.

        다음의 retrieved context를 이용하여 질문에 답하세요.
        무조건 한글로만 답해야 합니다


        #Previous Chat History : {chat_history}
        #Question : {question}
        #Context : {context}
        #Answer : 
    """
)

c = docload(d_path, embedding_model_name)
d = c.get_dir(glob = '**/*.txt', loader_cls = TextLoader, silent_errors = False, loader_kwargs = {'autodetect_encoding':True})
t = c.split_text(d, chunk_size = 50, chunk_overlap = 0)
print(t)



vec = vectordb(embedding, t)
#db = vec.init_db(distance_strategy = distance_strategy)
#db = vec.db_save(v_path, db)
db2 = vec.db_load(path = v_path)

basic_ret = vec.db_ret(db2, 5)
bm25 = vec.bm_ret(t, 5)
ensemble = vec.ensemble_ret([basic_ret, bm25], [0.5, 0.5], 5)

en_test = ensemble.invoke("고객번호 1번은 카페에서 얼마나 썼어?")
en_test2 = basic_ret.invoke("고객번호 1번은 카페에서 얼마나 썼어?")
en_test3 = bm25.invoke("고객번호 1번은 카페에서 얼마나 썼어?")
print(en_test)
print(en_test2)
print(en_test3)



chain = rag_chain(llm, prompt, basic_ret, session_ids, store)

account_chain = chain.get_rag_history()
result = account_chain.invoke({"question":"내가 카페에 쓴 금액이 총 얼마인지 알려줘"}, config = {"configurable" : {"session_id" : session_ids}})
print(result)
result2 = account_chain.invoke({"question":"내가 서점에 쓴 금액을 알려줘"}, config = {"configurable" : {"session_id" : session_ids}})
print(result2)
result3 = account_chain.invoke({"question":"그러면 내가 서점에 많이썼어? 아니면 카페에 많이썼어?"}, config = {"configurable" : {"session_id" : session_ids}})
print(result3)
