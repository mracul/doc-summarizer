import os
from typing import List, Dict, Any
from camel.toolkits import BaseToolkit, GithubToolkit
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import hashlib
import re
from dotenv import load_dotenv
from io import BytesIO
from rank_bm25 import BM25Okapi
import numpy as np

load_dotenv()

class IngestionToolkit(BaseToolkit):
    """A toolkit for ingesting data from various sources."""

    def _parse_github_url(self, url: str) -> tuple[str, str]:
        """Parses a GitHub URL to extract the repository name and directory path."""
        try:
            # Split the URL to get the part after 'github.com/'
            path_part = url.split("github.com/")[1]
            
            # Split the path into components
            parts = path_part.split('/')
            
            # The first two parts are the owner and the repo name
            if len(parts) < 2:
                raise ValueError("URL does not contain owner and repository name.")
            repo_name = f"{parts[0]}/{parts[1]}"
            
            # Check if a directory path is specified
            directory_path = ""
            if 'tree' in parts:
                tree_index = parts.index('tree')
                # The path is everything after the branch name (which is after 'tree')
                if len(parts) > tree_index + 2:
                    directory_path = "/".join(parts[tree_index + 2:])
            
            return repo_name, directory_path
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid GitHub URL format. Could not parse: {e}")

    def load_from_path(self, path: str, logger: callable = print) -> List[Dict[str, Any]]:
        """
        Loads data from a given path, which can be a file, directory, or GitHub URL.
        
        Args:
            path (str): The path to the data source.
            logger (callable): The function to use for logging. Defaults to print.
            
        Returns:
            List[Dict[str, Any]]: A list of documents, where each document is a dictionary
                                 containing 'file_path' and 'elements'.
        """
        docs = []
        logger(f"Attempting to load from path: {path}", "info")

        if "github.com" in path:
            logger("Detected GitHub URL. Cloning repository...", "info")
            repo_name, directory_path = self._parse_github_url(path)
            github_token = os.getenv("GITHUB_TOKEN")
            github_toolkit = GithubToolkit(repo_name, access_token=github_token)
            file_paths = github_toolkit.get_all_file_paths(directory_path)
            logger(f"Found {len(file_paths)} files in repository.", "info")
            for file_path in file_paths:
                content = github_toolkit.retrieve_file_content(file_path)
                try:
                    # Use BytesIO to treat the content as a file
                    elements = partition(file=BytesIO(content.encode()), file_filename=file_path)
                    docs.append({"file_path": file_path, "elements": elements})
                    logger(f"Successfully processed and added: {file_path}", "success")
                except Exception as e:
                    logger(f"Error processing file {file_path} from GitHub: {e}", "error")
            return docs

        if os.path.isdir(path):
            logger("Detected directory path. Walking through files...", "info")
            file_count = 0
            for root, _, files in os.walk(path):
                file_count += len(files)
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        elements = partition(filename=file_path)
                        docs.append({"file_path": file_path, "elements": elements})
                        logger(f"Successfully processed and added: {file_path}", "success")
                    except Exception as e:
                        logger(f"Error processing file {file_path}: {e}", "error")
            logger(f"Found and attempted to process {file_count} files.", "info")
        elif os.path.isfile(path):
            logger("Detected single file path.", "info")
            try:
                elements = partition(filename=path)
                docs.append({"file_path": path, "elements": elements})
                logger(f"Successfully processed and added: {path}", "success")
            except Exception as e:
                logger(f"Error processing file {path}: {e}", "error")
        else:
            logger(f"Path {path} is not a valid file, directory, or GitHub URL.", "error")
            
        return docs

class IndexingToolkit(BaseToolkit):
    """A toolkit for indexing documents, including chunking, embedding, and storing."""

    def __init__(self, qdrant_client: QdrantClient, collection_name: str) -> None:
        """
        Initializes the IndexingToolkit.

        Args:
            qdrant_client (QdrantClient): An instance of the Qdrant client.
            collection_name (str): The name of the collection to use.
        """
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def chunk_document(self, document: Dict[str, Any]) -> List[str]:
        """
        Chunks a document based on its file type using structure-aware strategies.

        Args:
            document (Dict[str, Any]): The document to chunk, containing 'file_path' and 'elements'.

        Returns:
            List[str]: A list of chunked text.
        """
        file_path = document.get("file_path", "")
        elements = document.get("elements", [])
        
        if file_path.endswith(".py"):
            # Basic AST chunking could be implemented here
            # For now, we'll use a simple text-based approach for code
            return [elem.text for elem in elements]
        elif file_path.endswith((".md", ".rst")):
            return [chunk.text for chunk in chunk_by_title(elements)]
        elif file_path.endswith(".ipynb"):
            # Each cell becomes a chunk
            return [elem.text for elem in elements]
        else:
            # Default chunking for other file types
            return [chunk.text for chunk in chunk_by_title(elements)]

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Embeds a list of text chunks using a SentenceTransformer model.
        
        Args:
            chunks (List[str]): The text chunks to embed.
            
        Returns:
            List[List[float]]: A list of embeddings.
        """
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
        return embeddings.tolist()

    def deduplicate_and_store(self, chunks: List[str], embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]) -> None:
        """
        Deduplicates chunks and stores them in a Qdrant vector database.
        Uses the hash of the chunk content as the point ID for efficient upserting.
        
        Args:
            chunks (List[str]): The text chunks.
            embeddings (List[List[float]]): The embeddings of the chunks.
            metadata_list (List[Dict[str, Any]]): A list of metadata dicts corresponding to each chunk.
        """
        points = []
        for i, (chunk, metadata) in enumerate(zip(chunks, metadata_list)):
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            
            points.append(
                models.PointStruct(
                    id=chunk_hash,
                    vector=embeddings[i],
                    payload={"text": chunk, **metadata},
                )
            )
        
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

class HybridRetriever:
    """
    Performs hybrid retrieval using semantic search and BM25 keyword search,
    then reranks the results.
    """

    def __init__(self, qdrant_client: QdrantClient, embedding_model: SentenceTransformer, collection_name: str):
        """
        Initializes the HybridRetriever.

        Args:
            qdrant_client (QdrantClient): An instance of the Qdrant client.
            embedding_model (SentenceTransformer): The model to use for embeddings.
            collection_name (str): The name of the collection to use.
        """
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name

    def retrieve(self, semantic_query: str, keyword_query: str, metadata_filter: Dict = None, bm25_weight: float = 0.5, top_k: int = 100, final_top_m: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves and reranks documents using a hybrid approach.

        Args:
            semantic_query (str): The query for semantic vector search.
            keyword_query (str): The query for BM25 keyword search.
            metadata_filter (Dict, optional): Qdrant metadata filter. Defaults to None.
            bm25_weight (float, optional): The weight to give to BM25 scores. Defaults to 0.5.
            top_k (int, optional): The number of initial results to fetch. Defaults to 100.
            final_top_m (int, optional): The final number of results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A list of reranked documents with their payloads.
        """
        query_embedding = self.embedding_model.encode(semantic_query, convert_to_tensor=False).tolist()
        
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=metadata_filter,
            with_payload=True
        )
        
        if not search_results:
            return []

        candidates = [hit.payload for hit in search_results]
        candidate_texts = [doc['text'] for doc in candidates]
        
        # Normalize semantic scores
        semantic_scores = np.array([hit.score for hit in search_results])
        if len(semantic_scores) > 1:
            semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-10)
        else:
            semantic_scores = np.array([1.0])


        # Calculate BM25 scores
        tokenized_candidates = [doc.split() for doc in candidate_texts]
        bm25 = BM25Okapi(tokenized_candidates)
        query_tokens = keyword_query.split()
        bm25_scores = bm25.get_scores(query_tokens)
        if len(bm25_scores) > 1:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        else:
            bm25_scores = np.array([1.0])


        # Combine scores and rerank
        combined_scores = (1 - bm25_weight) * semantic_scores + bm25_weight * bm25_scores
        top_indices = combined_scores.argsort()[::-1][:final_top_m]
        
        return [candidates[i] for i in top_indices]
