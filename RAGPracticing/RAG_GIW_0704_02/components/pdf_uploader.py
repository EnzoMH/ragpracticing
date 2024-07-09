import streamlit as st
from utils.file_handlers import save_uploaded_file

def pdf_uploader():
    uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type="pdf")
    if uploaded_file is not None:
        try:
            st.success(f"파일 '{uploaded_file.name}'이(가) 업로드되었습니다.")
            save_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"파일 저장 중 오류가 발생했습니다: {str(e)}")
    return uploaded_file