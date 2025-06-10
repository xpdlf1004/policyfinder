import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import faiss
import json
from pathlib import Path
from backend.models.schema import Policy

class PolicyEmbedder:
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.policies_path = Path("data/policy_data.json")
        self.index_path = Path("data/policy.index")
        
    def create_embeddings(self, policies: List[Policy]) -> np.ndarray:
        """Create embeddings for policy texts."""
        texts = [f"{p.candidate} {p.topic} {p.text}" for p in policies]
        return self.model.encode(texts, convert_to_numpy=True)
    
    def build_index(self, policies: List[Policy], embeddings: np.ndarray):
        """Build FAISS index from embeddings with policy IDs as metadata."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Save policy IDs as metadata
        self.policy_ids = [p.id for p in policies]
        
    def save_index(self):
        """Save FAISS index and policy IDs."""
        if self.index is None:
            raise ValueError("No index to save")
            
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save policy IDs
        with open("data/policy_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.policy_ids, f)
        
    def load_index(self) -> bool:
        """Load existing FAISS index and policy IDs if available."""
        if not self.index_path.exists() or not Path("data/policy_ids.json").exists():
            return False
            
        self.index = faiss.read_index(str(self.index_path))
        with open("data/policy_ids.json", "r", encoding="utf-8") as f:
            self.policy_ids = json.load(f)
        return True
        
    def search(self, query: str, k: int = 5) -> List[int]:
        """Search for similar policies using query and return policy IDs."""
        if self.index is None:
            raise ValueError("Index not initialized")
            
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, k)
        
        return [self.policy_ids[i] for i in indices[0]] 