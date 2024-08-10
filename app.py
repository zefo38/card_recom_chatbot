import streamlit as st
from DocLoader import docload
from RAG_VectorDB import vectordb
from RAGChain import rag_chain
import pandas as pd
import numpy as np
import langchain
import random
import time
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import TextLoader


llm = OllamaLLM(model = 'llama3:8b', temperature = 0.0)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embedding_model_name = 'intfloat/multilingual-e5-large'
embedding = HuggingFaceEmbeddings(model_name = embedding_model_name, model_kwargs = {'device' : 'cpu'}, encode_kwargs = {"normalize_embeddings" : True})
store = {}
session_ids = 'test1'


d_path = './customer_id_1'
v_path = './Faiss_DB'
c = docload(d_path, embedding_model_name)
d = c.get_dir(glob = '**/*.txt', loader_cls = TextLoader, silent_errors = False, loader_kwargs = {'autodetect_encoding':True})
t = c.split_text(d, chunk_size = 50, chunk_overlap = 0)
vec = vectordb(embedding, t)


db_loaded = vec.db_load(path = v_path)

basic_ret = vec.db_ret(db_loaded, 10)
bm25 = vec.bm_ret(t, 10)
ensemble = vec.ensemble_ret([basic_ret, bm25], [0.8, 0.2], 10)
chain = rag_chain(llm, ensemble, session_ids, store)
account_chain = chain.get_rag_history()

st.title("Simple Chat")

def get_response(user_input):
    responses = account_chain.invoke({"question": user_input}, config={"configurable": {"session_id": session_ids}})
    st.write(responses)
    if isinstance(responses, str):
        response_text = responses
    else:
        response_text = responses.get("output_text", "응답을 처리하는 중 문제가 발생했습니다.")
    return response_text

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)