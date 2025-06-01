import os
from typing import List
from openai import OpenAI
from anthropic import Anthropic
from ..models.schema import Policy

class PolicyGenerator:
    def __init__(self, model_type: str = "openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
    def generate_response(self, query: str, policies: List[Policy]) -> str:
        """Generate response using LLM based on retrieved policies."""
        context = self._format_context(policies)
        
        if self.model_type == "openai":
            return self._generate_openai(query, context)
        else:
            return self._generate_claude(query, context)
            
    def _format_context(self, policies: List[Policy]) -> str:
        """Format policies into context string."""
        context = "관련 공약 정보:\n\n"
        for i, policy in enumerate(policies, 1):
            context += f"{i}. {policy.candidate}의 {policy.topic} 공약:\n"
            context += f"{policy.text}\n\n"
        return context
        
    def _generate_openai(self, query: str, context: str) -> str:
        """Generate response using OpenAI GPT."""
        prompt = f"""다음은 대통령 후보들의 공약 정보입니다. 주어진 질문에 대해 공약 정보를 바탕으로 답변해주세요.

질문: {query}

{context}

답변:"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 대통령 후보들의 공약을 분석하고 설명하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    def _generate_claude(self, query: str, context: str) -> str:
        """Generate response using Claude."""
        prompt = f"""다음은 대통령 후보들의 공약 정보입니다. 주어진 질문에 대해 공약 정보를 바탕으로 답변해주세요.

질문: {query}

{context}"""
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.7,
            system="당신은 대통령 후보들의 공약을 분석하고 설명하는 전문가입니다.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text 