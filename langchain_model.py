import langchain
import transformers
from DocLoader import docload
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
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




class chat_chain():
    def __init__(self, llm, memory, retriever):
        self.llm = llm
        self.memory = memory
        self.retriever = retriever

    def get_chain_ordinary(self, prom):
        prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE + prom
        o_chain = prompt | self.llm | StrOutputParser()
        return o_chain
    
    def get_chain_account(self):
        prompt = PromptTemplate(
        input_variables=["chat_history", "query"],

            template=f"""
            당신은 가계부, 카드 추천 기능도 하는 챗봇입니다
            이 데이터는 날짜, 고객, 그 고객의 소비에 관한 데이터 입니다.
            당신은 무조건 한글로만 답해야 합니다
            금액의 단위는 원입니다
            질문자의 고객번호는 1번 입니다.
            데이터를 제대로 보고 알려주세요. 이모티콘은 없어도 되지만 정확도는 높아야 합니다

            
            {{chat_history}}
            Human : {{query}}
            AI:
            """
        )
        question_prompt = PromptTemplate(
            input_variables=["chat_history", "query"],
            template = f"""
            질문자의 고객번호는 1번입니다
            대화를 바탕으로 다음 질문을 생성하세요
            무조건 한글로만 해야 합니다
            대화 : {{chat_history}}
            질문 : {{query}}
            """
        )

        ac_chain = LLMChain(prompt = prompt, llm = self.llm, memory = self.memory)
        qa_chain = load_qa_chain(llm = self.llm, chain_type = 'stuff')
        question_gen_chain = LLMChain(prompt = question_prompt, llm = self.llm)
        conv_chain = ConversationalRetrievalChain(retriever = self.retriever, combine_docs_chain = qa_chain, question_generator = question_gen_chain , memory = self.memory)
        return conv_chain
    
    def get_chain_recsys(self):
        prompt = ChatPromptTemplate.from_template([
            ("system", "소비내역과 카드 정보, 카드 추천 시스템을 바탕으로 혜택이 큰 카드를 추천해줍니다"),
            ("user", "{user_input}")
        ])
        rec_chain = prompt | self.llm | StrOutputParser()
        return rec_chain
    
    def chat_rec(self, input):
        rec_chat = self.get_chain_recsys()
        recommendation = self.rec_model(input)
        out = rec_chat.invoke({"user" : input, "recommendations" : recommendation})
        return out
    
    def save_memory(self, input_text, output_text):
        if not isinstance(input_text, str):
            raise ValueError("input_text must be a string")
        if not isinstance(output_text, str):
            raise ValueError("output_text must be a string")

        context = self.memory.save_context({"input": input_text}, {"output": output_text})
        return context