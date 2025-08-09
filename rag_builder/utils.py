import json
import os
from typing import Dict, Any

from rag_builder.config import RAGS_CONFIG_PATH

def read_rags_config() -> Dict[str, Any]:
    """Reads the RAG configurations from the JSON file."""
    if not os.path.exists(RAGS_CONFIG_PATH):
        return {"rags": [], "active_rag": None}
    with open(RAGS_CONFIG_PATH, "r") as f:
        return json.load(f)

def write_rags_config(config: Dict[str, Any]):
    """Writes the RAG configurations to the JSON file."""
    with open(RAGS_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def get_active_rag(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the active RAG configuration."""
    active_rag_name = config.get("active_rag")
    if not active_rag_name:
        return None
    for rag in config.get("rags", []):
        if rag.get("name") == active_rag_name:
            return rag
    return None

def set_active_rag(config: Dict[str, Any], rag_name: str) -> bool:
    """Sets the active RAG in the configuration."""
    for rag in config.get("rags", []):
        if rag.get("name") == rag_name:
            config["active_rag"] = rag_name
            write_rags_config(config)
            return True
    return False