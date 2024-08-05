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
from langchain.memory import ConversationEntityMemory
import faiss
from langchain_community.docstore import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoModelForCausalLM
from langchain_ollama.llms import OllamaLLM



llm = OllamaLLM(model = 'llama3:8b')
embedding_model_name = "intfloat/multilingual-e5-large"

path = './source4'
v_path = './vectordb'
data = './processed_file(1) (9).csv'

embedding = HuggingFaceEmbeddings(model_name = embedding_model_name, model_kwargs = {'device' : 'cpu'}, encode_kwargs = {"normalize_embeddings" : True})
distance_strategy = DistanceStrategy.COSINE
memory = ConversationEntityMemory(llm = llm)
text = '친구처럼 대화해줘'
text_prompt = ChatPromptTemplate.from_template(text)

c = docload(path, embedding_model_name)
d = c.get_dir(glob = '**/*.tsv', loader_cls = TextLoader, silent_errors = False, loader_kwargs = {'autodetect_encoding':True})
t = c.split_text(d, chunk_size = 200, chunk_overlap = 50)


chain = chat_chain(llm, memory)
account_chain = chain.get_chain_account(data_path = data)
account_chain.invoke('고객번호가 1번인 고객이 카페에 쓴 금액을 알려줘')