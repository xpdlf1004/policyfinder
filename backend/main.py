from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from dotenv import load_dotenv
import json
from fastapi.responses import HTMLResponse
from typing import List

# Set environment variable to disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .models.schema import Question, PolicyResponse
from .data_loader import DataLoader
from .rag.embed import PolicyEmbedder
from .rag.retrieve import PolicyRetriever
from .rag.generate import ResponseGenerator

# Load environment variables
load_dotenv()

app = FastAPI(title="PolicyFinder")

# Setup templates
templates = Jinja2Templates(directory="backend/templates")

# Initialize components
data_loader = DataLoader()
embedder = PolicyEmbedder()
retriever = PolicyRetriever(embedder)
faiss_generator = ResponseGenerator(use_qdrant=False)
qdrant_generator = ResponseGenerator(use_qdrant=True)

# Load index on startup
@app.on_event("startup")
async def startup_event():
    if not embedder.load_index():
        print("Warning: No FAISS index found. Please run 'python script/embed_policies.py' first.")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with search interface."""
    # Load policy data to get unique candidates and topics
    with open("data/policy_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        candidates = sorted(set(p["candidate"] for p in data))
        topics = sorted(set(p["topic"] for p in data))
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "candidates": candidates,
            "topics": topics
        }
    )

@app.post("/ask")
async def ask_question(question: Question) -> PolicyResponse:
    """Process question and return response with sources."""
    try:
        # 검색 엔진에 따라 다른 처리
        if question.search_engine == "qdrant":
            print("Qdrant 검색 엔진 사용")
            # Qdrant 검색 파라미터 설정
            search_params = {
                "k": 5,  # 상위 5개 결과
                "score_threshold": 0.7  # 유사도 임계값
            }
            
            # Qdrant 검색 실행
            policies = qdrant_generator.qdrant_pipeline.run_pledge_query_with_sources(
                question.question,
                candidate_filter=question.candidate_filter,
                topic_filter=question.topic_filter,
                **search_params
            )
            
            # 검색 결과가 있는 경우
            if policies:
                # 컨텍스트 생성
                context = qdrant_generator.qdrant_pipeline._create_context_from_policies(policies)
                
                # 응답 생성
                answer = qdrant_generator.qdrant_pipeline.llm.invoke(
                    qdrant_generator.qdrant_pipeline.prompt.format(
                        context=context,
                        input=question.question
                    )
                )
                
                # FAISS와 동일한 형식으로 응답 반환
                return PolicyResponse(
                    answer=answer.content,  # AIMessage에서 content 추출
                    sources=policies
                )
            else:
                return PolicyResponse(answer="검색 조건에 맞는 공약을 찾을 수 없습니다. 다른 검색어나 필터를 사용해보세요.", sources=[])
        else:
            # FAISS를 사용하는 경우 (기존 로직)
            policies = retriever.retrieve(
                question.question,
                candidate_filter=question.candidate_filter,
                topic_filter=question.topic_filter
            )
            answer, referenced_policies = faiss_generator.generate_response(question.question, policies)
            return PolicyResponse(answer=answer, sources=referenced_policies)
    except Exception as e:
        print(f"질문 처리 중 오류 발생: {str(e)}")
        return PolicyResponse(
            answer="죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            sources=[]
        )

@app.get("/candidates")
async def get_candidates(search_engine: str = "faiss") -> List[str]:
    """Get list of candidates."""
    try:
        if search_engine == "qdrant":
            return qdrant_generator.qdrant_pipeline.get_candidates()
        return faiss_generator.retriever.get_candidates()
    except Exception as e:
        print(f"후보 목록 가져오기 오류: {str(e)}")
        return []

@app.get("/topics")
async def get_topics(search_engine: str = "faiss") -> List[str]:
    """Get list of topics."""
    try:
        if search_engine == "qdrant":
            return qdrant_generator.qdrant_pipeline.get_topics()
        return faiss_generator.retriever.get_topics()
    except Exception as e:
        print(f"주제 목록 가져오기 오류: {str(e)}")
        return [] 