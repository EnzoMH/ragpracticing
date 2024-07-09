# 파일 및 정규표현식 관련 모듈
import os  # 운영 체제와 상호작용하기 위한 모듈
import re  # 정규 표현식을 사용하기 위한 모듈

# PyTorch 관련 모듈
import torch  # PyTorch 프레임워크

# Transformers 관련 모듈
from transformers import (
    AutoModelForCausalLM,  # 언어 모델을 로드하는 함수
    AutoTokenizer,  # 토크나이저를 로드하는 함수
    BitsAndBytesConfig,  # 양자화 및 압축을 위한 설정
    TrainingArguments,  # 모델 훈련을 위한 인자 설정
    pipeline,  # 파이프라인을 생성하는 함수
    logging,  # 로깅을 위한 함수
    AutoConfig,  # 설정을 로드하는 함수
    GenerationConfig  # 텍스트 생성을 위한 설정
)

# Langchain 관련 모듈
from langchain_community.document_loaders import PyPDFLoader  # PDF 문서 로더
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트 분할기

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain.schema import Document  # 문서 스키마
from langchain_community.vectorstores import Chroma  # 벡터 저장소
from langchain.chains.question_answering import load_qa_chain  # 질의 응답 체인 로드
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain  # 질의 응답 체인
from langchain.memory import ConversationSummaryMemory  # Langchain 메모리 관련 모듈대화 요약 메모리
from langchain.text_splitter import CharacterTextSplitter  # Langchain 텍스트 분할기 모듈 문자 기반 텍스트 분할기
from langchain.prompts import PromptTemplate  # Langchain 프롬프트 템플릿 관련 모듈 프롬프트 템플릿
import huggingface_hub # huggingface 접속하기 위한 모듈

#from langchain.embeddings import HuggingFaceEmbeddings  # 임베딩 생성기
#from langchain.llms import HuggingFacePipeline  # Langchain LLM 관련 모듈 HuggingFace 파이프라인을 사용하는 LLM

#huggingface_hub.login('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')
huggingface_hub.login('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')

modelName = "meta-llama/Meta-Llama-3-8B-Instruct" # 모델을 잘 가져다 쓰는게 중요, 물론 내가 학습한 AI 모델을 적용하는게 좋기는 함
revision= "main"

config = AutoConfig.from_pretrained(modelName, revision=revision)
model = AutoModelForCausalLM.from_pretrained(modelName, revision=revision, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(modelName, revision=revision)
# print(model)

sourceName = "50_게임전시회.pdf"

loader = PyPDFLoader(f'{sourceName}')
documents = loader.load()

output = []
# text 정제
for page in documents:
    text = page.page_content
    output.append(text)

doc_chunks = []
for line in output:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # 최대 청크 길이
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], #  텍스트를 청크로 분할하는 데 사용되는 문자 목록
        chunk_overlap=100, # 인접한 청크 간에 중복되는 문자 수
    )
    chunks = text_splitter.split_text(line)
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk, metadata={ "source": f"{sourceName}", "page": i}
        )
        doc_chunks.append(doc)
        
# HuggingFaceEmbeddings 클래스를 사용하여 임베딩 모델 로드
embed_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-nli",  # 한국어 문장 임베딩 모델
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

# Chroma 데이터베이스 디렉토리 및 컬렉션 이름 설정
persist_directory = "./chromadb"
collection_name = "hkcode_documents"

# 기존의 Chroma 인덱스를 로드하거나 없으면 새로 생성
try:
    index = Chroma(
        embedding_function=embed_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    # 컬렉션이 존재하면 문서 추가
    index.add_documents(doc_chunks)
except Exception as e:
    print(f"Error loading collection: {e}")
    # 컬렉션이 없으면 새로 생성하여 문서 추가
    index = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embed_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

# 변경된 내용을 디스크에 저장
index.persist()

# 검색기를 생성하여 유사도 검색 수행
retriever = index.as_retriever(search_kwargs={"k": 1})  # 검색 결과 상위 1개 반환

# 모델 설정을 로드하여 GenerationConfig 객체 생성
gen_cfg = GenerationConfig.from_pretrained(modelName)
# 생성 설정 세부 조정
gen_cfg.max_new_tokens = 1024  # 생성할 최대 토큰 수
gen_cfg.temperature = 0.0000001  # 텍스트 생성의 무작위성을 낮추는 온도 설정
gen_cfg.return_full_text = True  # 생성된 텍스트 전체 반환
gen_cfg.do_sample = True  # 샘플링을 통해 텍스트 생성
gen_cfg.repetition_penalty = 1.11  # 반복 페널티 설정

# HuggingFace의 파이프라인을 사용하여 텍스트 생성 파이프라인 생성
pipe = pipeline(
    task="text-generation",  # 파이프라인 작업 유형 설정
    model=model,  # 사전 훈련된 언어 모델
    tokenizer=tokenizer,  # 모델에 맞는 토크나이저
    generation_config=gen_cfg,  # 생성 설정 적용
    device="cpu"  # 사용할 디바이스 설정 (기본 GPU를 사용할 경우 0으로 설정)
)

# LangChain의 HuggingFacePipeline을 사용하여 LLM 객체 생성
llm = HuggingFacePipeline(pipeline=pipe)
# LLAMA 2 LLM에 권장되는 프롬프트 스타일을 사용합니다.
prompt_template = """
[INST] <>
You are a wise assistant. Answer questions accurately in Korean
<>
{context}

Question: {question} [/INST]
"""

# 프롬프트 템플릿을 생성합니다.
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# RetrievalQA 체인을 생성합니다.
Chain_pdf = RetrievalQA.from_chain_type(
    llm=llm,  # 사용할 LLM을 지정합니다.
    chain_type="stuff",  # 체인의 타입을 지정합니다.
    retriever=retriever,  # 문서를 검색할 때 사용할 retriever를 지정합니다.
    return_source_documents=True, #소스출서 확인 지정
    chain_type_kwargs={"prompt": prompt},  # 체인 타입에 대한 추가 인자를 지정합니다.
)

def extract_text_after_inst(text):
    answer = text['result'].strip()
    marker = '[/INST]'
    start = answer.find(marker)
    if start != -1:
        # Extract text after the marker
        return answer[start + len(marker):].strip()
    else:
        # Return an empty string or a suitable message if marker not found
        return "Marker '[/INST]' not found in the answer."
    
# 질의 내용 설정
query = "해당 제안요청서의 제안서 장수 제한은 몇 페이지야?"

# 질의에 대한 답변을 생성합니다.
result = Chain_pdf.invoke(query)
print(result)
answer2 = extract_text_after_inst(result)
print(answer2)

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app=FastAPI(
    title="Python hkcode Assistent",
    version="1.0",
    decsription="API Server"
)

from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# FastAPI 연계 확인
class Input(BaseModel):
    input : str
    
@app.post("/predict", status_code=200)
async def predictDl(x: Input):
    print(x)
    prompt = x.input
    result = Chain_pdf.invoke(prompt)
    resul2 = extract_text_after_inst(result)
    return {"prediction":resul2}
@app.get("/")
async def root():
    return {"message":"onine"}

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)