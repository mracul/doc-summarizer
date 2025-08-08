import json
from typing import List, Dict, Any

from rag_builder.config import RAGS_CONFIG_PATH

def read_rags_config() -> List[Dict[str, Any]]:
    """Reads the RAG configurations from the JSON file."""
    if not os.path.exists(RAGS_CONFIG_PATH):
        return []
    with open(RAGS_CONFIG_PATH, "r") as f:
        return json.load(f)

def write_rags_config(config: List[Dict[str, Any]]):
    """Writes the RAG configurations to the JSON file."""
    with open(RAGS_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)