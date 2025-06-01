from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from dotenv import load_dotenv

from .models.schema import Question, PolicyResponse
from .data_loader import DataLoader
from .rag.embed import PolicyEmbedder
from .rag.retrieve import PolicyRetriever
from .rag.generate import PolicyGenerator

# Load environment variables
load_dotenv()

app = FastAPI(title="PolicyFinder")

# Setup templates
templates = Jinja2Templates(directory="backend/templates")

# Initialize components
data_loader = DataLoader()
embedder = PolicyEmbedder()
retriever = PolicyRetriever(embedder)
generator = PolicyGenerator(model_type=os.getenv("LLM_PROVIDER", "openai"))

# Load data and build index on startup
@app.on_event("startup")
async def startup_event():
    try:
        policies = data_loader.load_data()
        if not embedder.load_index():
            embeddings = embedder.create_embeddings([p.dict() for p in policies])
            embedder.build_index([p.dict() for p in policies], embeddings)
            embedder.save_index()
    except FileNotFoundError:
        print("Warning: No policy data found. Please add data/policy_data.json")

@app.get("/")
async def home(request: Request):
    """Render home page with search interface."""
    candidates = data_loader.get_candidates()
    topics = data_loader.get_topics()
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
    # Retrieve relevant policies
    policies = retriever.retrieve(
        question.question,
        candidate_filter=question.candidate_filter,
        topic_filter=question.topic_filter
    )
    
    # Generate response
    answer = generator.generate_response(question.question, policies)
    
    return PolicyResponse(answer=answer, sources=policies) 