import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from rag.embed import PolicyEmbedder

def main():
    print("정책 임베딩 및 인덱스 생성 시작...")
    data_loader = DataLoader()
    policies = data_loader.load_data()
    embedder = PolicyEmbedder()
    embeddings = embedder.create_embeddings([p.dict() for p in policies])
    embedder.build_index([p.dict() for p in policies], embeddings)
    embedder.save_index()
    print("FAISS 인덱스가 data/policy.index에 저장되었습니다.")

if __name__ == "__main__":
    main() 