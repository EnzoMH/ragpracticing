from glob import glob
from langchain_community.document_loaders import PyMuPDFLoader

# PDF 파일들의 경로를 찾기 위해 glob 패턴을 사용하여 파일 리스트를 가져옴
pdf_files = glob("C:\\Users\\qksso\\ai_project\\rag\\*.pdf")

# PyMuPDFLoader 인스턴스 생성
loader = PyMuPDFLoader()

# 각 PDF 파일을 순회하면서 페이지 수 출력
for pdf_file in pdf_files:
    # 각 PDF 파일에 대해 load 메서드를 호출하여 데이터를 가져옴
    document = loader.load(pdf_file)

    # 페이지 수 출력
    num_pages = len(document.pages)
    print(f"{pdf_file}: 페이지 수 = {num_pages}")
