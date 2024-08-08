import langchain
import transformers
from DocLoader import docload
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
import langchain_core
import langchain_text_splitters
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.agents import AgentType
from langchain.memory import ConversationEntityMemory
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from RAG_VectorDB import vectordb
from DocLoader import docload
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough




class chat_chain():
    def __init__(self, llm, memory, retriever):
        self.llm = llm
        self.memory = memory
        self.retriever = retriever
    

    
    def get_chain_account(self):
        prompt = ChatPromptTemplate.from_messages(
        [("system","""
            당신은 가계부 역할과 카드 추천 역할을 하는 챗봇입니다.
            나의 고객번호는 1번 입니다.
            고객번호 1번 이외의 데이터는 조회하면 안됩니다.
            당신은 무조건 한글로만 답해야 합니다.
            금액의 단위는 원입니다.

            context를 사용하여 다음 질문에 답변해 주세요:
            {context}
            """,
        ),
        ("human", "{question}")])
        rag_chain = ConversationChain({"context" : self.retriever, "question" : RunnablePassthrough(),} | prompt | self.llm)

        return rag_chain