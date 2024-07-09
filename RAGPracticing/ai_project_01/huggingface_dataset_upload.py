from huggingface_hub import login

login(token="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")

import os
from datasets import Dataset, DatasetDict, Features, Value

# PDF 파일 경로 리스트 생성
pdf_dir = r"C:\\Users\\qksso\\Downloads\\dataset\\40"
pdf_files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir) if file.endswith('.pdf')]

# 데이터셋 생성
data = {'file_paths': pdf_files}
dataset = Dataset.from_dict(data)

# 데이터셋 피쳐 정의 (optional)
features = Features({'file_paths': Value('string')})
dataset = Dataset.from_dict(data, features=features)

# 데이터셋을 DatasetDict로 래핑 (optional, 필요에 따라)
dataset_dict = DatasetDict({'train': dataset})

# Hugging Face Hub에 업로드
dataset_dict.push_to_hub("DrugHugBug/pdf", private=True)
