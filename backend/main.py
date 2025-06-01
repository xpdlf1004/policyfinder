from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from dotenv import load_dotenv
import json
from fastapi.responses import HTMLResponse

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
generator = ResponseGenerator()

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
    # Retrieve relevant policies
    policies = retriever.retrieve(
        question.question,
        candidate_filter=question.candidate_filter,
        topic_filter=question.topic_filter
    )
    # Generate response and get referenced policies
    answer, referenced_policies = generator.generate_response(question.question, policies)
    return PolicyResponse(answer=answer, sources=referenced_policies) 