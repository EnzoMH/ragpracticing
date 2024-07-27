from openai import OpenAI

client = OpenAI(  
    base_url="http://sionic.chat:8001/v1",     
    api_key=""
)

response = client.chat.completions.create(     
    model="xionic-ko-llama-3-70b",     
    messages=[         
        {"role": "system", 
         "content": "You are an AI assistant. You will be given a task. You must generate a detailed and long answer in korean."
        },         
        {"role": "user", 
         "content": "아이브 리더가 활동했던 이전 걸그룹의 명칭은?"
        }
    ] 
)

print(response.choices[0].message.content)
