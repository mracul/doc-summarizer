import os
import asyncio
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()
from camel.models import ModelFactory
from camel.messages import BaseMessage
from camel.types import ModelPlatformType

from rag_builder.toolkits import IngestionToolkit, IndexingToolkit, HybridRetriever
from rag_builder.agents import SynthesisAgent, ClarificationAgent
from rag_builder.config import VECTOR_STORAGE_PATH

class RAGPipeline:
    """
    Manages the entire RAG pipeline, from ingestion to querying.
    This class orchestrates the workforce of agents and their tools.
    """

    def __init__(self) -> None:
        """
        Initializes the RAGPipeline, setting up necessary clients and agents.
        """
        self.qdrant_client: QdrantClient = QdrantClient(path=VECTOR_STORAGE_PATH)

    async def async_init(self) -> None:
        """
        Asynchronously initializes the models and agents.
        """
        await self._initialize_models()
        self._initialize_agents()

    async def _initialize_models(self) -> None:
        """Initializes the language models required for the agents."""
        self.synthesis_model = await asyncio.to_thread(
            ModelFactory.create,
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="openai/gpt-oss-20b:free",
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict={"max_tokens": 4096},
        )
        self.clarification_model = await asyncio.to_thread(
            ModelFactory.create,
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="openai/gpt-oss-20b:free",
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict={"max_tokens": 1024, "temperature": 0.2},
        )

    def _initialize_agents(self) -> None:
        """Initializes the agents that form the workforce."""
        self.synthesis_agent: SynthesisAgent = SynthesisAgent(model=self.synthesis_model)
        self.clarification_agent: ClarificationAgent = ClarificationAgent(model=self.clarification_model)

    def ingest(self, path: str, collection_name: str, logger: callable = print) -> bool:
        """
        Ingests data from the given path into the specified collection.

        Args:
            path (str): The path to the data source (file, directory, or Git URL).
            collection_name (str): The name of the Qdrant collection.
            logger (callable): The function to use for logging. Defaults to print.
        
        Returns:
            bool: True if ingestion was successful, False otherwise.
        """
        logger(f"Starting ingestion for collection: {collection_name}")
        
        ingestion_toolkit = IngestionToolkit()
        indexing_toolkit = IndexingToolkit(self.qdrant_client, collection_name)

        try:
            logger("Loading documents...")
            documents = ingestion_toolkit.load_from_path(path, logger=logger)
            if not documents:
                logger("No documents found to ingest.", "warning")
                return True  # Not a failure, just nothing to do.

            all_chunks = []
            all_metadata = []

            logger("Chunking documents...")
            for doc in documents:
                logger(f"Processing document: {doc.get('file_path', 'unknown')}")
                chunks = indexing_toolkit.chunk_document(doc)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "file_path": doc.get("file_path", ""),
                        "chunk_id": i,
                        "source_type": os.path.splitext(doc.get("file_path", ""))[1],
                        "tags": [],  # Placeholder for future tag implementation
                        "commit_hash": "",  # Placeholder for future git integration
                    })

            if all_chunks:
                logger("Embedding chunks...")
                embeddings = indexing_toolkit.embed_chunks(all_chunks)
                logger("Deduplicating and storing chunks...")
                indexing_toolkit.deduplicate_and_store(all_chunks, embeddings, all_metadata)

            logger("Data ingestion completed successfully.", "success")
            return True
        except Exception as e:
            logger(f"An error occurred during ingestion: {e}", "error")
            return False

    async def ask(self, query: str, collection_name: str, metadata_filter: dict = None, logger: callable = print):
        """
        Asks a question to the RAG pipeline using hybrid retrieval.

        Args:
            query (str): The question to ask.
            collection_name (str): The collection to query against.
            metadata_filter (dict, optional): Metadata filter for Qdrant. Defaults to None.
            logger (callable): The function to use for logging. Defaults to print.
        """
        logger(f"Processing query for collection: {collection_name}")

        # Clarification step
        logger("Clarifying query and extracting search terms...")
        clarification_msg = BaseMessage.make_user_message(role_name="User", content=query)
        clarification_response = await self.clarification_agent.step(clarification_msg)
        
        try:
            # The result from the agent might be in a code block
            json_str = clarification_response.content.replace("```json\n", "").replace("\n```", "")
            clarification_data = json.loads(json_str)
            
            refined_query = clarification_data['refined_query_for_embedding']
            search_terms = clarification_data['search_terms']
            keyword_query = " ".join(search_terms)
            
            logger(f"Intent: {clarification_data.get('intent', 'N/A')}", "info")
            logger(f"Refined Query for Embedding: {refined_query}", "info")
            logger(f"Search Terms: {', '.join(search_terms)}", "info")

        except (json.JSONDecodeError, KeyError) as e:
            logger(f"Could not parse clarification agent output: {e}", "error")
            logger("Falling back to original query.", "warning")
            refined_query = query
            keyword_query = query

        # Initialize the retriever and embedding model for this task
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        retriever = HybridRetriever(self.qdrant_client, embedding_model, collection_name)

        # Perform hybrid retrieval
        logger("Retrieving relevant chunks...")
        retrieved_chunks = retriever.retrieve(
            semantic_query=refined_query,
            keyword_query=keyword_query,
            metadata_filter=metadata_filter,
            top_k=200,
            final_top_m=20
        )

        if not retrieved_chunks:
            logger("No relevant context found.", "warning")
            logger("Answer:", "success")
            logger("I could not find any relevant information to answer your question.")
            return

        # Build the detailed prompt for the synthesis agent
        logger("Building prompt for synthesis agent...")
        chunk_texts = []
        for chunk in retrieved_chunks:
            chunk_texts.append(f"""
Chunk ID: {chunk.get('chunk_id', 'N/A')}
File Path: {chunk.get('file_path', 'N/A')}
Source Type: {chunk.get('source_type', 'N/A')}
Tags: {chunk.get('tags', [])}

Content:
{chunk.get('text', '')}
""")
        context_str = "\n---\n".join(chunk_texts)
        
        final_prompt = f"""You are an expert assistant answering user queries based strictly on the provided context.

Each context chunk includes metadata: file path, source type, tags, and chunk ID.

Use this information to cite your sources in the answer explicitly by referencing file paths and tags.

Do NOT answer beyond the provided context.

---

Context chunks:
{context_str}

Question:
{refined_query}

Answer with references to the source chunks:"""
        # Ask the synthesis agent
        synthesis_msg = BaseMessage.make_user_message(role_name="User", content=final_prompt)
        synthesis_response = await self.synthesis_agent.step(synthesis_msg)
        logger("Synthesis complete.")
        
        logger("Answer:", "success")
        logger(synthesis_response.content)

    def get_collection_stats(self, collection_name: str) -> dict:
        """
        Retrieves statistics for a given collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            dict: A dictionary containing collection statistics.
        """
        try:
            collection_info = self.qdrant_client.get_collection(collection_name=collection_name)
            return {
                "documents": collection_info.points_count,
                "vectors": "384-dim",  # Assuming all-MiniLM-L6-v2
            }
        except Exception:
            return {"documents": "N/A", "vectors": "N/A"}
