# card_chatbot
![langchain](https://github.com/user-attachments/assets/196777a2-540d-4b20-9086-da5b21c808e4)

![jumpstart-fm-rag](https://github.com/user-attachments/assets/4a8d5e08-9f15-4b49-8ad0-cc6eb3ddb704)


## 랭체인을 활용한 가계부 챗봇 및 카드 추천 시스템

### 활용 모델 : llama3

### 개요
- 랭체인을 활용해 QnA 형식의 챗봇을 구축
- 기존 모델을 활용하면서 가계부 데이터, 카드 혜택 정보, 추천 시스템 코드는 RAG를 활용하여 모델이 사용할 수 있게 함
- 이를 위해 가계부 데이터, 카드 혜택 정보, 추천 시스템 코드는 벡터화 후 벡터DB에 저장
- retriever로 벡터DB에서 질문과 유사한 문서들을 찾아낸 후 답변 생성

### 상세 내용
- 벡터화를 위해 임베딩 모델은 "intfloat/multilingual-e5-large"를 사용
- 데이터 로드할 때는 DirectoryLoader클래스 이용
  - 추후에 추가될 수 있는 다른 형식의 파일(.txt, .tsv)들도 효과적으로 로드할 수 있게 하기 위함
  - 파이썬 코드는 다른 클래스를 별도로 사용함
- 벡터 DB : FAISS
  - 오픈소스 데이터베이스
  - 대규모 벡터 데이터베이스에서 유사도 검색이 빠름

- 데이터 전처리
  - RAG를 사용하기 위해 정형 데이터를 텍스트 데이터로 변환
    - 정형 데이터를 바로 벡터화 했을 땐 성능이 제대로 나오지 않았음
    - csv2txt 클래스를 만들어서 사용
  - 텍스트 데이터로 변환 후 벡터화 -> 벡터DB에 저장

- RAG
  - retriever의 경우 ensemble 방식을 사용
    - 최대한 자원을 덜 소비하면서 검색 성능을 높이기 위함
    - ensemble에 들어가는 retriever의 경우 vectorstore retriever(search_type : mmr)과 bm_25를 사용
  - VectorDB
    - 벡터DB하나 당 한 명의 정보만 들어감
      - 그래야만 더 개인화된 서비스 구현이 가능
      - 개인정보 유출 문제 방지도 가능해짐
      - 또한 각 DB마다 카드 정보와 카드 추천 시스템 관련 정보도 함께 들어있음


- langchain
  - retriever를 사용하면서 이전 대화 내용도 기억하게 해야함
    - 그러기 위해서 대화 내용을 store에 저장 후 프롬프트를 통해 활용하게 함

### 코드
- DocLoader.py
  - 데이터 로드 및 split, embedding 메소드가 포함된 클래스
  - 이 클래스를 통해 위의 작업들을 모듈화

- RAG_VectorDB.py
  - 벡터DB 생성, Retriever 생성, DB저장, 로드, merge 메소드가 있음
  - Retriever는 BM_25, vectorstore retriever, ensemble 세가지 생성 가능

- RAGChain.py
  - 체인을 생성
  - 또한 세션 히스토리를 저장해 챗봇이 활용 가능하게 해줌

- CSV2TXT.py
  - csv파일을 텍스트 파일로 변환
  - 고객 가계부 데이터를 텍스트로 변환해줌