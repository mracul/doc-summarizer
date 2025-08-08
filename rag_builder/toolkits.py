import os
from typing import List, Dict, Any
from camel.toolkits import BaseToolkit, function_tool
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient, models
import hashlib

class IngestionToolkit(BaseToolkit):
    """A toolkit for ingesting data from various sources."""

    @function_tool
    def load_from_path(self, path: str) -> List[Dict[str, Any]]:
        """
        Loads data from a given path, which can be a file, directory, or Git repository.
        
        Args:
            path (str): The path to the data source.
            
        Returns:
            List[Dict[str, Any]]: A list of documents, where each document is a dictionary.
        """
        docs = []
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        elements = partition(filename=file_path)
                        docs.append({"file_path": file_path, "elements": elements})
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
        elif os.path.isfile(path):
            try:
                elements = partition(filename=path)
                docs.append({"file_path": path, "elements": elements})
            except Exception as e:
                print(f"Error processing file {path}: {e}")
        else:
            raise ValueError(f"Path {path} is not a valid file or directory.")
            
        return docs

class IndexingToolkit(BaseToolkit):
    """A toolkit for indexing documents, including chunking, embedding, and storing."""

    def __init__(self, qdrant_client: QdrantClient, collection_name: str):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.ollama_embeddings = OllamaEmbeddings(model="mistral")

    @function_tool
    def chunk_document(self, document: Dict[str, Any]) -> List[str]:
        """
        Chunks a document into smaller pieces.
        
        Args:
            document (Dict[str, Any]): The document to chunk.
            
        Returns:
            List[str]: A list of chunked text.
        """
        elements = document.get("elements", [])
        chunks = chunk_by_title(elements)
        return [chunk.text for chunk in chunks]

    @function_tool
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Embeds a list of text chunks using a local Ollama instance.
        
        Args:
            chunks (List[str]): The text chunks to embed.
            
        Returns:
            List[List[float]]: A list of embeddings.
        """
        return self.ollama_embeddings.embed_documents(chunks)

    @function_tool
    def deduplicate_and_store(self, chunks: List[str], embeddings: List[List[float]], document_path: str) -> None:
        """
        Deduplicates chunks and stores them in a Qdrant vector database.
        
        Args:
            chunks (List[str]): The text chunks.
            embeddings (List[List[float]]): The embeddings of the chunks.
            document_path (str): The path to the original document.
        """
        points = []
        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            
            # Check for existence of the hash
            response = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="hash",
                            match=models.MatchValue(value=chunk_hash),
                        )
                    ]
                ),
                limit=1,
            )
            
            if not response[0]:
                points.append(
                    models.PointStruct(
                        id=f"{document_path}_{i}",
                        vector=embeddings[i],
                        payload={"text": chunk, "source": document_path, "hash": chunk_hash},
                    )
                )
        
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
