import os
import re
from openai import OpenAI
from typing import List, Tuple, Dict, Any
from ..models.schema import Policy
from ..qdrant_rag.qdrant_rag_pipeline import QdrantRAGPipeline

class ResponseGenerator:
    def __init__(self, use_qdrant: bool = False):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.use_qdrant = use_qdrant
        if use_qdrant:
            self.qdrant_pipeline = QdrantRAGPipeline()

    def format_context(self, policies: List[Policy]) -> str:
        """Format retrieved policies into context string."""
        # Group policies by candidate
        policies_by_candidate = {}
        for policy in policies:
            if policy.candidate not in policies_by_candidate:
                policies_by_candidate[policy.candidate] = []
            policies_by_candidate[policy.candidate].append(policy)
        
        # Format context
        context = "관련 공약 정보:\n\n"
        for policy in policies:
            context += f"[공약 ID: {policy.id}] - {policy.candidate}의 공약\n"
            context += f"주제: {policy.topic}\n"
            context += f"내용: {policy.text}\n"
            context += f"출처: {policy.source}\n\n"
        return context

    def extract_referenced_policy_ids(self, text: str) -> List[int]:
        """Extract policy IDs referenced in the text."""
        # Find all matches of [공약: 숫자] pattern
        matches = re.findall(r'\[공약:\s*(\d+)\]', text)
        # Convert to integers and remove duplicates
        return list(set(int(id) for id in matches))

    def generate_response(self, question: str, policies: List[Policy]) -> Tuple[str, List[Policy]]:
        """Generate response using OpenAI API and return referenced policies."""
        if not policies:
            return "죄송합니다. 검색 조건에 맞는 공약을 찾을 수 없습니다. 다른 검색어나 필터를 사용해보세요.", []
            
        # Format context from policies
        context = self.format_context(policies)
        
        # Create prompt
        prompt = f"""다음은 대선 후보들의 공약 정보입니다:

{context}

위 정보를 바탕으로 다음 질문에 답변해주세요:
{question}

주의사항:
- 주어진 공약 정보만을 사용하여 답변해주세요.
- 정보가 없는 내용은 지어내지 마세요.
- 확실하지 않은 내용은 언급하지 마세요.
- 답변에서 참고한 공약은 [공약: ID] 형식으로 표시해주세요. 예: "김철수의 주거 공약에 따르면 [공약: 1]..."

답변:"""
        
        # Generate response
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 대선 후보들의 공약을 분석하고 비교하는 전문가입니다. 주어진 정보만을 사용하여 정확하고 객관적인 답변을 제공해주세요. 답변에는 참고한 공약 ID [공약: 숫자]를 표시해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4096
        )
        
        answer = response.choices[0].message.content
        
        # Extract referenced policy IDs
        referenced_ids = self.extract_referenced_policy_ids(answer)
        
        # Filter policies to only include referenced ones
        referenced_policies = [p for p in policies if p.id in referenced_ids]
        
        return answer, referenced_policies 