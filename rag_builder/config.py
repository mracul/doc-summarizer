
import os

# Path to the directory where data files are stored.
DATA_PATH = "rag_builder/data"

# Path to the directory where the vector storage (Qdrant) will be created.
VECTOR_STORAGE_PATH = "rag_builder/vector_storage"

# Path to the JSON file that stores the RAG configurations.
RAGS_CONFIG_PATH = "rag_builder/rags.json"

# The embedding model to use.
EMBEDDING_MODEL = "mistral-embed"

# The chat model to use for generating answers.
CHAT_MODEL = "mistral-large-latest"

# The OCR model to use.
OCR_MODEL = "mistral-large-latest"
