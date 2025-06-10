from pydantic import BaseModel
from typing import List, Optional

class Policy(BaseModel):
    id: int
    candidate: str
    topic: str
    text: str
    source: str

class PolicyResponse(BaseModel):
    answer: str
    sources: List[Policy]

class Question(BaseModel):
    question: str
    candidate_filter: Optional[str] = None
    topic_filter: Optional[str] = None
    search_engine: str = "faiss"  # 기본값은 faiss 