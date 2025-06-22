from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from ..models.schema import Policy
import json

load_dotenv()

class QdrantRAGPipeline:
    def __init__(self):
        # 초기 설정
        self.collection_name = "policy_collection"
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # 프롬프트 템플릿 정의
        self.prompt_template = (
            "다음은 대통령 후보들의 공약 내용입니다:\n\n"
            "{context}\n\n"
            "이 내용을 바탕으로 사용자 질문에 답하세요:\n\n"
            "질문: {input}"
        )

    def _create_search_params(
        self,
        query: str,
        candidate_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        k: int = 5,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """검색 파라미터를 생성합니다."""
        # 검색 필터 생성
        search_filter = None
        if candidate_filter or topic_filter:
            filter_conditions = []
            if candidate_filter:
                filter_conditions.append({
                    "key": "candidate",
                    "match": {"value": candidate_filter}
                })
            if topic_filter:
                filter_conditions.append({
                    "key": "topic",
                    "match": {"value": topic_filter}
                })
            search_filter = {
                "must": filter_conditions
            }
        
        return {
            "limit": k,
            "score_threshold": score_threshold,
            "query_filter": search_filter
        }

    def _format_search_results(self, search_results: List[Dict]) -> List[Policy]:
        """검색 결과를 Policy 객체로 변환합니다."""
        policies = []
        for result in search_results:
            payload = result.payload
            policies.append(Policy(
                id=int(payload.get("id", 0)),
                candidate=payload.get("candidate", ""),
                topic=payload.get("topic", ""),
                text=payload.get("pledge", ""),
                source=payload.get("source", "")
            ))
        return policies

    def _create_context_from_policies(self, policies: List[Policy]) -> str:
        """Policy 객체들로부터 컨텍스트를 생성합니다."""
        return "\n\n".join([
            f"[공약 ID: {p.id}] - {p.candidate}의 공약\n"
            f"주제: {p.topic}\n"
            f"내용: {p.text}\n"
            f"출처: {p.source}"
            for p in policies
        ])

    def _embed_query(self, query: str) -> List[float]:
        """OpenAI API를 사용하여 쿼리를 임베딩합니다."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {str(e)}")
            return []

    def run_pledge_query_with_sources(
        self,
        query: str,
        candidate_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        k: int = 5,
        score_threshold: float = 0.7
    ) -> List[Policy]:
        """질문에 대한 검색을 실행하고 Policy 객체 리스트를 반환합니다."""
        try:
            print(f"=== Debug Info ===")
            print(f"Query: {query}")
            print(f"Candidate filter: {candidate_filter}")
            print(f"Topic filter: {topic_filter}")
            print(f"Search parameters: k={k}, score_threshold={score_threshold}")

            # 쿼리 임베딩 생성
            query_vector = self._embed_query(query)
            if not query_vector:
                print("임베딩 생성 실패")
                return []

            # 검색 파라미터 설정
            search_params = self._create_search_params(
                query=query,
                candidate_filter=candidate_filter,
                topic_filter=topic_filter,
                k=k,
                score_threshold=score_threshold
            )
            
            # 검색 실행
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                **search_params
            )
            
            print(f"검색 결과 수: {len(search_results)}")
            
            # 검색 결과를 Policy 객체로 변환
            policies = []
            for result in search_results:
                try:
                    # 메타데이터에서 정책 정보 가져오기
                    payload = result.payload
                    
                    policy = Policy(
                        id=int(payload.get("id", 0)),
                        candidate=payload.get("candidate", ""),
                        topic=payload.get("topic", ""),
                        text=payload.get("pledge", ""),
                        source=payload.get("source", "")
                    )
                    policies.append(policy)
                    print(f"정책 변환 성공: {policy.id}")
                except Exception as e:
                    print(f"정책 변환 오류: {str(e)}")
                    print(f"페이로드: {result.payload}")
                    continue
            
            return policies
            
        except Exception as e:
            print(f"Qdrant 검색 중 오류 발생: {str(e)}")
            return []

    def get_candidates(self) -> List[str]:
        """Qdrant에서 모든 후보 목록을 가져옵니다."""
        try:
            # Qdrant에서 모든 문서의 candidate 필드 값을 가져옴
            response = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=1000,  # 충분히 큰 수
                with_payload=True,
                with_vectors=False
            )
            
            # 고유한 candidate 값 추출
            candidates = set()
            for point in response[0]:
                if point.payload:
                    candidate = point.payload.get("candidate")
                    if candidate:
                        candidates.add(candidate)
            
            return sorted(list(candidates))
        except Exception as e:
            print(f"후보 목록 가져오기 실패: {str(e)}")
            return []

    def get_topics(self) -> List[str]:
        """Qdrant에서 모든 주제 목록을 가져옵니다."""
        try:
            # Qdrant에서 모든 문서의 topic 필드 값을 가져옴
            response = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=1000,  # 충분히 큰 수
                with_payload=True,
                with_vectors=False
            )
            
            # 고유한 topic 값 추출
            topics = set()
            for point in response[0]:
                if point.payload:
                    topic = point.payload.get("topic")
                    if topic:
                        topics.add(topic)
            
            return sorted(list(topics))
        except Exception as e:
            print(f"주제 목록 가져오기 실패: {str(e)}")
            return []

    def llm(self):
        """LLM 응답을 생성하는 메서드"""
        return self.openai_client.chat.completions.create

    def prompt(self):
        """프롬프트 템플릿을 반환하는 메서드"""
        return self.prompt_template 