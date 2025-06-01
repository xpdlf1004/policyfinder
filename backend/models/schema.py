from pydantic import BaseModel
from typing import List, Optional

class Policy(BaseModel):
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