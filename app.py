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
from PIL import Image

import cluster
import recommend

st.set_page_config(layout="wide")

st.title("💬 KB Chatbot")
st.caption("더 나은 금융생활을 위한 맞춤형 서비스를 지원해드립니다.")

tab1, tab2 = st.tabs(['챗봇', '카드 추천'])

llm = OllamaLLM(model='llama3:8b', temperature=0.0)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding_model_name = 'intfloat/multilingual-e5-large'
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'}, encode_kwargs={"normalize_embeddings": True})
store = {}
session_ids = 'test1'

d_path = './customer_id_1'
v_path = './Faiss_DB'
c = docload(d_path, embedding_model_name)
d = c.get_dir(glob='**/*.txt', loader_cls=TextLoader, silent_errors=False, loader_kwargs={'autodetect_encoding': True})
t = c.split_text(d, chunk_size=50, chunk_overlap=0)
vec = vectordb(embedding, t)

db_loaded = vec.db_load(path=v_path)

basic_ret = vec.db_ret(db_loaded, 10)
bm25 = vec.bm_ret(t, 10)
ensemble = vec.ensemble_ret([basic_ret, bm25], [0.8, 0.2], 10)
chain = rag_chain(llm, ensemble, session_ids, store)
account_chain = chain.get_rag_history()

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

# 챗봇 탭
with tab1:
    st.write("챗봇 인터페이스")

# 카드 추천
with tab2:
    st.subheader(f'국민 님에게 적합한 카드 추천')
    st.markdown('')
    st.markdown(f'#### 지난 달 국민님 소비 분석 결과')
    st.write('')
    cards1, bene = recommend.data()

    img1 = Image.open(f'card\Img\{cards1[0]}.png')
    img1 = img1.resize((255, 150))

    img2 = Image.open(f'card\Img\{cards1[1]}.png')
    img2 = img2.resize((255, 150))

    img3 = Image.open(f'card\Img\{cards1[2]}.png')
    img3 = img3.resize((255, 150))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img1)
        st.markdown(f'##### **1. {cards1[0]}**')
        st.markdown(f'**{bene[0].values[0]}**, **{bene[0].values[1]}**, **{bene[0].values[3]}**에서 혜택 특화!')

    with col2:
        st.image(img2)
        st.markdown(f'##### **2. {cards1[1]}**')
        st.markdown(f'**{bene[1].values[0]}**, **{bene[1].values[1]}**, **{bene[1].values[2]}**에서 혜택 특화!')

    with col3:
        st.image(img3)
        st.markdown(f'##### **3. {cards1[2]}**')
        st.markdown(f'**{bene[2].values[0]}**, **{bene[2].values[1]}**, **{bene[2].values[2]}**에서 혜택 특화!')

    st.subheader('')
    st.markdown('#### **나랑 비슷한 소비패턴을 가진 사람들은 어떤 카드를 쓸까?**')

    if st.button('확인해보기'):
        card2 = cluster.card_recommend()

        for i in range(len(card2)):
            if '다담' in card2[i]:
                card2[i] = '다담'
            elif 'MyWESH' in card2[i]:
                card2[i] = 'MyWESH'
            elif 'EasyAll티타늄' in card2[i]:
                card2[i] = 'EasyAll티타늄'

        img1 = Image.open(f'card\Img\{card2[0]}.png')
        img1 = img1.resize((255, 150))

        img2 = Image.open(f'card\Img\{card2[1]}.png')
        img2 = img2.resize((255, 150))

        img3 = Image.open(f'card\Img\{card2[2]}.png')
        img3 = img3.resize((255, 150))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img1)
            st.markdown(f' **1. {card2[0]}**')
        with col2:
            st.image(img2)
            st.markdown(f'##### **2. {card2[1]}**')
        with col3:
            st.image(img3)
            st.markdown(f'##### **3. {card2[2]}**')
