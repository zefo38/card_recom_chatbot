import streamlit as st
from DocLoader import docload
from RAG_VectorDB import vectordb
from RAGChain import rag_chain
import pandas as pd
import numpy as np
import langchain


st.title("account & recsys chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "질문을 적어 주세요 무엇을 도와 드릴까요?"}]


