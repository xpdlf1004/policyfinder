# PolicyFinder (정책파인더)

대통령 후보 공약 정보를 질문 기반으로 검색, 요약, 비교하는 RAG 시스템입니다.

## 주요 기능

- 질문 기반 공약 검색
- 관련 공약 요약 및 비교
- 후보별 공약 필터링
- 주제별 공약 필터링

## 기술 스택

- Backend: Python (FastAPI)
- Frontend: HTML Templates (Jinja2)
- Vector Search: FAISS
- Embedding: BGE-KO / KoSimCSE
- LLM: OpenAI GPT / Claude

## 설치 및 실행

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 정책 임베딩 및 인덱스 생성:
```bash
python script/embed_policies.py
```
- 정책 데이터(`data/policy_data.json`)를 수정한 경우, 위 스크립트를 다시 실행해야 합니다.

3. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일에 API 키 등 설정
```

4. 서버 실행:
```bash
uvicorn backend.main:app --reload
```

5. 브라우저에서 접속:
```
http://localhost:8000
```

## 프로젝트 구조

```
policyfinder/
├── backend/
│   ├── main.py                # FastAPI 엔트리포인트
│   ├── rag/
│   │   ├── embed.py          # JSON 기반 텍스트 임베딩 처리
│   │   ├── retrieve.py       # 유사 공약 검색 (FAISS)
│   │   └── generate.py       # LLM을 통한 답변 생성
│   ├── models/
│   │   └── schema.py         # Pydantic 모델 정의
│   ├── data_loader.py        # JSON 로딩 및 초기 인덱싱
│   └── templates/            # HTML 템플릿 (Jinja2)
│       └── index.html        # 질문 입력 및 응답 출력 페이지
├── data/
│   ├── policy_data.json      # 사용자가 제공한 공약 JSON
│   └── policy.index          # FAISS 인덱스 파일
├── script/
│   └── embed_policies.py     # 임베딩 및 인덱스 생성 스크립트
└── README.md
```

## 데이터 형식

공약 데이터는 다음과 같은 JSON 형식을 사용합니다:

```json
{
  "candidate": "후보명",
  "topic": "정책 분야",
  "text": "공약 본문",
  "source": "공약 출처 URL"
}
```

## 라이선스

MIT License 