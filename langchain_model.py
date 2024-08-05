import langchain
import transformers
from DocLoader import docload
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
import langchain_core
import langchain_text_splitters
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.agents import AgentType
from langchain.memory import ConversationEntityMemory
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from RAG_VectorDB import vectordb
from DocLoader import docload
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE


llm = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
path2 = './source2'
path1 = './sorce'
encoding = 'utf-8'
source_column = '고객번호'
c = docload()
r = vectordb()

class chat_chain():
    def __init__(self, llm, memory, rec_model):
        self.llm = llm
        self.memory = memory
        self.rec_model = rec_model

    def get_chain_ordinary(self):
        text = '친구처럼 대화해줘'
        prompt = PromptTemplate.from_template(text)
        o_chain = prompt | self.llm | self.memory
        return o_chain
    
    def get_chain_account(self):
        prompt = ChatPromptTemplate.from_template([
            ("system", "소비내역을 참고해서 질문에 대답할 수 있습니다"),
            ("user", "{user_input}")
        ])
        ac_chain = prompt | self.llm | self.memory
        return ac_chain
    
    def get_chain_recsys(self):
        prompt = ChatPromptTemplate.from_template([
            ("system", "소비내역과 카드 정보, 카드 추천 시스템을 바탕으로 혜택이 큰 카드를 추천해줍니다"),
            ("user", "{user_input}")
        ])
        rec_chain = prompt | self.llm | self.memory
        return rec_chain
    
    def chat_rec(self, input):
        rec_chat = self.get_chain_recsys()
        recommendation = self.rec_model(input)
        out = rec_chat.invoke({"user" : input, "recommendations" : recommendation})
        return out