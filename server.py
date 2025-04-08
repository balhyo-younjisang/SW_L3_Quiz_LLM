import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
import random
import datetime
import json

app = FastAPI()

# Load FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index_path = "./faiss_index"
db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

class QueryRequest(BaseModel):
    section : str


def get_random_subject():
  subject = ["UI 테스트", "사용성 테스트 계획하기",  "테스트 기법 선정", "테스트 환경 구축", "사용성 테스트 계획서 작성", "사용성 테스트 수행하기", "사용성 테스트 수행", "평가 분석서 작성 및 이슈 도출", "테스트 결과 보고하기", "UI 개선방안 및 수정계획 수립", "UI 개선 결과보고서 공유"
             "개발자환경구축", "운영체제 기초 활용하기", "운영체제 식별", "운영체제 기본 명령어 활용", "운영체제 작업 우선순위 설정", "기본 개발환경 구축하기", "운영체제 설치", "개발도구 설치", "개발도구 활용",
             "애플리케이션배포", "애플리케이션 배포환경 구성하기", "애플리케이션 배포환경 구성", "애플리케이션 소스 검증하기", "애플리케이션 소스 검증", "애플리케이션 빌드하기", "애플리케이션 빌드", "애플리케이션 배포하기", "애플리케이션 배포",
             "애플리케이션테스트수행", "애플리케이션 테스트 수행하기", "테스트 수행", "결함관리", "애플리케이션 결함 조치하기", "조치 우선순위 결정", "결함 조치 관리"
             "응용SW기초기술활용", "네트워크 기초 활용하기", "네트워크 프로토콜 활용", "미들웨어 기초 활용하기", "미들웨어 파악", "미들웨어 운용", "데이터베이스 기초 활용하기", "데이터베이스 특징 식별", "관계형 데이터베이스 테이블 정의", "관계형 데이터베이스 테이블 조작"]
  return random.choice(subject)

@app.post("/query")
async def query(request : QueryRequest):
    question_section =  request.section  if request.section != None else get_random_subject()
    docs = db.similarity_search(question_section, k=5)
    context_parts = []
    sources = []
    for doc in docs:
        context_parts.append(doc.page_content)
        sources.append(doc.metadata.get("source", "알 수 없음"))

    context = "\n\n".join(context_parts)
    prompt = f"""
당신은 NCS 기반 정보처리산업기사 과정평가형 외부평가에 출제 경험이 있는 전문 출제자입니다.

아래 문서를 기반으로 실전 대비용 고난이도 문제 3개를 생성해주세요. 수험자는 고등학생 이상이며, 시험은 실제 자격증 평가와 동일한 수준으로 구성되어야 합니다.

✅ 문제 출제 조건:
1. 문제 유형은 다음 중 **2가지 이상 사용**해야 합니다:
   - 4지선다형 객관식 (정의/개념 기반)
   - 진위형
   - 정의 암기 또는 간단한 기술 설명
2. 각 문제에는 반드시 다음 항목이 포함되어 있습니다:
   - `type`: 문제 유형 (`객관식` / `진위형` / `단답형`)
   - `question`: 문제 본문
   - `options`: 보기 리스트 (객관식/진위형만; 단답형은 빈 리스트)
   - `answer`: 정답 (보기의 인덱스(0~3) 또는 직접 답안)
   - `explanation`: 해설 (개념 중심, 교육적 설명)
   - `difficulty`: 난이도 (`중` / `중상` / `상`)
3. 전체 문제 중 **2문제 이상은 주요 용어 또는 개념 정의**를 묻는 암기 기반 문제여야 합니다.
4. 보기에는 최소 1개 이상의 **혼란을 유도하는 유사 개념**을 포함하여, 단순 상식으로는 정답을 고르기 어렵도록 구성해야 합니다.

---  
**출력 형식** (JSON 배열):
```json
[  
  {{  
    "type": "객관식",  
    "question": "...",  
    "options": ["...", "...", "...", "..."],  
    "answer": 2,  
    "explanation": "...",  
    "difficulty": "중상"  
  }},  
  {{  
    "type": "단답형",  
    "question": "...",  
    "options": [],  
    "answer": "...",  
    "explanation": "...",  
    "difficulty": "상"  
  }},  
  {{  
    "type": "진위형",  
    "question": "...",  
    "options": ["O", "X"],  
    "answer": "O",  
    "explanation": "...",  
    "difficulty": "중"  
  }}  
]
[문서 내용]
{context} """

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": "자동 문제 생성",
        "context_excerpt": context[:300], 
        "questions": question_section,
        "sources": sources
    }
    
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/query_history.json", "a", encoding="utf-8") as f:
      f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
      
    
    response = llm.invoke([HumanMessage(content=prompt)])
    questions = response.content
    
    try:
        return json.loads(questions)
    except json.JSONDecodeError:
        return {"error": "JSON 디코딩 실패", "raw_response": questions}
