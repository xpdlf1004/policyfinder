import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 환경 변수 로드
load_dotenv()

def load_policy_data():
    """정책 데이터를 로드합니다."""
    with open('data/policy_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def create_qdrant_collection():
    """Qdrant 컬렉션을 생성합니다."""
    client = QdrantClient("localhost", port=6333)
    
    try:
        # 기존 컬렉션 삭제
        client.delete_collection(collection_name="policy_collection")
        print("기존 컬렉션이 삭제되었습니다.")
    except Exception as e:
        print(f"컬렉션 삭제 중 오류 (무시됨): {str(e)}")
    
    # 새 컬렉션 생성
    client.create_collection(
        collection_name="policy_collection",
        vectors_config=models.VectorParams(
            size=1536,  # OpenAI embeddings 크기
            distance=models.Distance.COSINE
        )
    )
    print("새 컬렉션이 생성되었습니다.")

def upload_to_qdrant():
    """정책 데이터를 Qdrant에 업로드합니다."""
    # OpenAI Embeddings 모델 초기화
    embeddings = OpenAIEmbeddings()
    
    # Qdrant 클라이언트 초기화
    client = QdrantClient("localhost", port=6333)
    
    # 컬렉션 재생성
    create_qdrant_collection()
    
    # 정책 데이터 로드
    policies = load_policy_data()
    print(f"로드된 정책 수: {len(policies)}")
    
    # 문서와 메타데이터 준비
    documents = []
    metadatas = []
    
    for policy in policies:
        # 문서 내용을 정책 텍스트로 설정
        content = policy["text"]
        documents.append(content)
        
        # 메타데이터 준비
        metadata = {
            "id": str(policy["id"]),  # id를 문자열로 변환
            "candidate": policy["candidate"],
            "topic": policy["topic"],
            "source": policy["source"],
            "pledge": policy["text"]  # text를 pledge로 저장
        }
        metadatas.append(metadata)
    
    # Qdrant에 업로드
    qdrant = Qdrant(
        client=client,
        collection_name="policy_collection",
        embeddings=embeddings
    )
    
    # 새 데이터 업로드
    qdrant.add_texts(
        texts=documents,
        metadatas=metadatas
    )
    
    print("데이터 업로드가 완료되었습니다.")

if __name__ == "__main__":
    try:
        upload_to_qdrant()
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}") 