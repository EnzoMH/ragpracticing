# Streamlit 기반 구동 모델, ENDPOINT 등의 문제 

import streamlit as st
import os
import time
import requests
# 경고 메시지 제거
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# userprompt
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# vectorDB
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# llms
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
# pdf loader
from langchain_community.document_loaders import PyPDFLoader
# pdf processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
# retrieval
from langchain.chains import RetrievalQA

# Set your Hugging Face API token here
HUGGINGFACEHUB_API_TOKEN = ""

# Set your model endpoint here
MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/your-username/your-model"

if not os.path.exists('pdfFiles'):
    os.makedirs('pdfFiles')

if not os.path.exists('vectorDB'):
    os.makedirs('vectorDB')

if 'template' not in st.session_state:
    st.session_state.template = """너는 학습하는 챗봇이야, 내가 하는 질문에 전문적이고 지식적으로 답변하길 바래
    Context: {context}
    History: {history}
    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='vectorDB',
                                          embedding_function=embedding_model)

def query_model(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {
        "inputs": prompt,
    }
    response = requests.post(MODEL_ENDPOINT, headers=headers, json=payload)
    return response.json()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Chatbot - to talk to PDFs")

uploaded_file = st.file_uploader("{PDF파일을 선택하세요}", type="pdf")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    st.text("파일 업로드가 되었습니다")
    if not os.path.exists('pdfFiles/' + uploaded_file.name):
        with st.status("파일 처리 중..."):
            bytes_data = uploaded_file.read()
            with open('pdfFiles/' + uploaded_file.name, 'wb') as f:
                f.write(bytes_data)

            loader = PyPDFLoader('pdfFiles/' + uploaded_file.name)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )

            all_splits = text_splitter.split_documents(data)

            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=embedding_model
            )

            st.session_state.vectorstore.persist()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=query_model,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                prompt = st.session_state.prompt.format(
                    context="",
                    history="",
                    question=user_input
                )
                response = query_model(prompt)
                result = response.get('generated_text', 'Error: No response')

            message_placeholder = st.empty()
            full_response = ""
            for chunk in result.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": result}
        st.session_state.chat_history.append(chatbot_message)

else:
    st.write("시작하시려면 PDF를 업로드해주세요")
