from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# 올라마 모델 Repository ID
repo_id = 'mistralai/Mistral-7B-v0.1'

# 사용자 질의
question = "Who is Son Heung Min?"

# 템플릿
template = """Question: {question}

# Answer: {answer}"""

# 프롬프트 템플릿 생성
prompt = PromptTemplate(template=template, input_variables=["question"])

# HuggingFaceHub 객체 생성
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.2, "max_length": 128}
)

# LLMChain 객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 실행
result = llm_chain.run(question=question)

# 결과 출력
print(result['answer'])