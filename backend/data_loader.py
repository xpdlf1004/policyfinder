import json
from pathlib import Path
from typing import List, Dict
from .models.schema import Policy

class DataLoader:
    def __init__(self, data_path: str = "data/policy_data.json"):
        self.data_path = Path(data_path)
        self.policies: List[Policy] = []
        
    def load_data(self) -> List[Policy]:
        """Load policy data from JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Policy data file not found at {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        self.policies = [Policy(**item) for item in raw_data]
        return self.policies
    
    def get_candidates(self) -> List[str]:
        """Get list of unique candidates."""
        return list(set(policy.candidate for policy in self.policies))
    
    def get_topics(self) -> List[str]:
        """Get list of unique topics."""
        return list(set(policy.topic for policy in self.policies))
    
    def filter_policies(self, candidate: str = None, topic: str = None) -> List[Policy]:
        """Filter policies by candidate and/or topic."""
        filtered = self.policies
        
        if candidate:
            filtered = [p for p in filtered if p.candidate == candidate]
        if topic:
            filtered = [p for p in filtered if p.topic == topic]
            
        return filtered 