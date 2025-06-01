import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import faiss
import os
from pathlib import Path

class PolicyEmbedder:
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.policies = []
        self.index_path = Path("data/policy.index")
        
    def create_embeddings(self, policies: List[Dict]) -> np.ndarray:
        """Create embeddings for policy texts."""
        texts = [f"{p['candidate']} {p['topic']} {p['text']}" for p in policies]
        return self.model.encode(texts, convert_to_numpy=True)
    
    def build_index(self, policies: List[Dict], embeddings: np.ndarray):
        """Build FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.policies = policies
        
    def save_index(self):
        """Save FAISS index and policy data."""
        if self.index is None:
            raise ValueError("No index to save")
            
        faiss.write_index(self.index, str(self.index_path))
        
    def load_index(self) -> bool:
        """Load existing FAISS index if available."""
        if not self.index_path.exists():
            return False
            
        self.index = faiss.read_index(str(self.index_path))
        return True
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar policies using query."""
        if self.index is None:
            raise ValueError("Index not initialized")
            
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, k)
        
        return [self.policies[i] for i in indices[0]] 