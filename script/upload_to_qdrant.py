import os
import json
from dotenv import load_dotenv
from openai import OpenAI
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

def get_embedding(text: str, client: OpenAI) -> list:
    """OpenAI API를 사용하여 텍스트의 임베딩을 생성합니다."""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"임베딩 생성 오류: {str(e)}")
        return None

def upload_to_qdrant():
    """정책 데이터를 Qdrant에 업로드합니다."""
    # OpenAI 클라이언트 초기화
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Qdrant 클라이언트 초기화
    qdrant_client = QdrantClient("localhost", port=6333)
    
    # 컬렉션 재생성
    create_qdrant_collection()
    
    # 정책 데이터 로드
    policies = load_policy_data()
    print(f"로드된 정책 수: {len(policies)}")
    
    # 각 정책을 Qdrant에 업로드
    for i, policy in enumerate(policies):
        try:
            # 텍스트 임베딩 생성
            embedding = get_embedding(policy["text"], openai_client)
            if embedding is None:
                print(f"정책 {policy['id']} 임베딩 생성 실패, 건너뜀")
                continue
            
            # Qdrant에 포인트 추가
            qdrant_client.upsert(
                collection_name="policy_collection",
                points=[
                    models.PointStruct(
                        id=policy["id"],
                        vector=embedding,
                        payload={
                            "id": str(policy["id"]),
                            "candidate": policy["candidate"],
                            "topic": policy["topic"],
                            "source": policy["source"],
                            "pledge": policy["text"]
                        }
                    )
                ]
            )
            
            if (i + 1) % 10 == 0:
                print(f"진행률: {i + 1}/{len(policies)}")
                
        except Exception as e:
            print(f"정책 {policy['id']} 업로드 오류: {str(e)}")
            continue
    
    print("데이터 업로드가 완료되었습니다.")

if __name__ == "__main__":
    try:
        upload_to_qdrant()
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}") 