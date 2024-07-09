import re
from langchain_community.document_loaders import PyMuPDFLoader

def extract_numbers_from_text(text):
    # 정규 표현식 패턴 설정
    number_pattern = r'\b\d+\b'  # 숫자 패턴: 연속된 숫자만 추출 (\b는 단어 경계)

    # 텍스트에서 숫자 추출
    numbers = re.findall(number_pattern, text)
    return numbers

# PyMuPDFLoader를 사용하여 PDF 파일 로드
loader = PyMuPDFLoader("50_국립생태원.pdf")

# load() 메서드를 사용하여 데이터 로드
data = loader.load()

# 각 페이지의 텍스트 데이터에서 숫자 추출하여 출력
for page in data:
    cleaned_text = page.page_content.replace("\n", "").replace("\t", "")
    numbers_on_page = extract_numbers_from_text(cleaned_text)
    print(f"페이지 {page.number}: 추출된 숫자들 - {numbers_on_page}")
