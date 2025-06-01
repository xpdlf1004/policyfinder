from typing import List, Dict
from .embed import PolicyEmbedder
from ..models.schema import Policy

class PolicyRetriever:
    def __init__(self, embedder: PolicyEmbedder):
        self.embedder = embedder
        
    def retrieve(self, query: str, k: int = 5, 
                candidate_filter: str = None, 
                topic_filter: str = None) -> List[Policy]:
        """Retrieve relevant policies based on query and filters."""
        # Get initial results
        results = self.embedder.search(query, k=k*2)  # Get more results for filtering
        
        # Apply filters
        if candidate_filter:
            results = [r for r in results if r['candidate'] == candidate_filter]
        if topic_filter:
            results = [r for r in results if r['topic'] == topic_filter]
            
        # Convert to Policy objects and limit to k results
        policies = [Policy(**r) for r in results[:k]]
        return policies
        
    def format_context(self, policies: List[Policy]) -> str:
        """Format retrieved policies into context string for LLM."""
        context = "관련 공약 정보:\n\n"
        for i, policy in enumerate(policies, 1):
            context += f"{i}. {policy.candidate}의 {policy.topic} 공약:\n"
            context += f"{policy.text}\n\n"
        return context 