from typing import List, Optional, Dict
import json
from pathlib import Path
from backend.models.schema import Policy
from .embed import PolicyEmbedder

class PolicyRetriever:
    def __init__(self, embedder: PolicyEmbedder):
        self.embedder = embedder
        self.policies_path = Path("data/policy_data.json")
        
    def _load_policies(self) -> Dict[int, Policy]:
        """Load all policies from JSON file."""
        with open(self.policies_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {p['id']: Policy(**p) for p in data}
        
    def retrieve(
        self,
        query: str,
        k: int = 5,
        candidate_filter: Optional[str] = None,
        topic_filter: Optional[str] = None
    ) -> List[Policy]:
        """Retrieve relevant policies based on query and filters."""
        # Get policy IDs from vector search
        policy_ids = self.embedder.search(query, k=k*2)  # Get more results for filtering
        
        # Load all policies
        policies = self._load_policies()
        
        # Get policies by ID and apply filters
        results = []
        for pid in policy_ids:
            policy = policies[pid]
            if candidate_filter and policy.candidate != candidate_filter:
                continue
            if topic_filter and policy.topic != topic_filter:
                continue
            results.append(policy)
            if len(results) >= k:
                break
                
        return results
        
    def format_context(self, policies: List[Policy]) -> str:
        """Format retrieved policies into context string for LLM."""
        context = "관련 공약 정보:\n\n"
        for i, policy in enumerate(policies, 1):
            context += f"{i}. {policy.candidate}의 {policy.topic} 공약:\n"
            context += f"{policy.text}\n\n"
        return context 