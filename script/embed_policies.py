import sys
from pathlib import Path
import json

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from backend.rag.embed import PolicyEmbedder
from backend.models.schema import Policy

def main():
    # Load policy data
    with open("data/policy_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        policies = [Policy(**p) for p in data]
    
    # Create embeddings
    embedder = PolicyEmbedder()
    embeddings = embedder.create_embeddings(policies)
    
    # Build and save index
    embedder.build_index(policies, embeddings)
    embedder.save_index()
    
    print("Embeddings created and saved successfully!")

if __name__ == "__main__":
    main() 