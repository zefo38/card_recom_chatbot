from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from DocLoader import docload
from RAG_VectorDB import vectordb

class rag_chain():
    def __init__(self, llm, prompt, retriever, session_ids, store):
        self.llm = llm
        self.prompt = prompt
        self.retriever = retriever
        self.session_ids = session_ids
        self.store = store
        self.chain = ({"context" : itemgetter("question") | self.retriever, "question" : itemgetter("question"), "chat_history" : itemgetter("chat_history"),} | self.prompt | self.llm | StrOutputParser())
    
    def get_session_history(self):
        print(f"[대화 세션 ID] : {self.session_ids}")
        if self.session_ids not in self.store:
            self.store[self.session_ids] = ChatMessageHistory()
        return self.store[self.session_ids]
    
    def get_rag_history(self):
        rag_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key = "question",
            history_messages_key = "chat_history",
        )
        return rag_with_history