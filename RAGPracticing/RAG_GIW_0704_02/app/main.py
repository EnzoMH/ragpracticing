import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from components.pdf_uploader import pdf_uploader
from components.chat_interface import chat_interface

def main():
    st.title("Chatbot - to talk to PDFs")
    
    uploaded_file = pdf_uploader()
    
    if uploaded_file is not None:
        chat_interface(uploaded_file)
    else:
        st.write("시작하시려면 PDF를 업로드해주세요")

if __name__ == "__main__":
    main()