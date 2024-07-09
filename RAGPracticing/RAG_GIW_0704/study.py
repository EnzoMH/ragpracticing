import streamlit as st 
# 다양한 어플리케이션을 만들 수 있는 파이썬 프레임워크; streamlit, 실행시 Webpage URL로 이동
# 간단한 코드로 대화형 웹 애플리케이셔늘 만들 수 있음
# 다양한 UI 컴포넌트를 제공, 사용자와의 상호작용을 쉽게 구현
# 실시간 데이터 업데이트 및 시각화 지원

import os
import time

#사용자 프롬프트

from langchain.prompts import PromptTemplate # 프롬프트 템플릿을 정의하고 관리할 수 있는 클래스
from langchain.memory import ConversationBufferMemory # 대화내역을 버퍼에 저장, 관리하는 메모리 클래스, 대화상태를 유지하며 이전 대화를 기억, 대화 히스토리를 관리하여 문맥을 이해하는데 도움 

# 벡터 데이터베이스(VectorDB)

from langchain_community.vectorstores import Chroma # 벡터 스토어를 관리하는 라이브러리
from langchain_community.embeddings.ollama import OllamaEmbeddings # 텍스트 데이터를 벡터로 변환하는 임베딩 클래스를 제공

# LLMs
from langchain_community.llms import Ollama # 대형 언어 모델(LLM)을 관리하고 사용할 수 있는 클래스
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # 스트리밍 방식으로 표준 출력을 처리하는 콜백 핸들러
from langchain.callbacks.manager import CallbackManager # 여러 콜백을 관리하고 실행 순서를 제어하는 클래스

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader #PDF 파일을 로드하고 처리하는 클래스

# PDF Processing
from langchain.text_splitter import RecursiveCharacterTextSplitter # 텍스트를 재귀적으로 분할하는 클래스 ## 왜 분할하는데 재귀적으로 분할할까?

# Retrieval
from langchain.chains import RetrievalQA # 정보 검색 기반 질의응답 체인을 구성하는 클래스
