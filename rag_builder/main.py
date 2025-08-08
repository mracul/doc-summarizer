import argparse
import uuid
import inquirer
from dotenv import load_dotenv
from camel.societies import Workforce
from camel.tasks import Task
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from qdrant_client import QdrantClient

from rag_builder.toolkits import IngestionToolkit, IndexingToolkit
from rag_builder.agents import ToolCriticAgent, RetrievalAgent, SynthesisAgent
from rag_builder.utils import read_rags_config, write_rags_config
from rag_builder.config import VECTOR_STORAGE_PATH

# Load environment variables from .env file
load_dotenv()

def create_rag(args):
    """Creates a new RAG index configuration."""
    print(f"Creating a new RAG index named '{args.name}'...")
    rags_config = read_rags_config()
    
    if any(rag['name'] == args.name for rag in rags_config):
        print(f"Error: A RAG index with the name '{args.name}' already exists.")
        return
        
    new_rag = {
        "name": args.name,
        "collection_name": str(uuid.uuid4()),
    }
    rags_config.append(new_rag)
    write_rags_config(rags_config)
    print(f"RAG index '{args.name}' created successfully.")

def list_rags(args):
    """Lists all available RAG indexes."""
    rags_config = read_rags_config()
    if not rags_config:
        print("No RAG indexes found.")
        return
    
    print("Available RAG indexes:")
    for rag in rags_config:
        print(f"- {rag['name']} (Collection: {rag['collection_name']})")

def ingest_data(args):
    """Ingests data into a specified RAG index."""
    rags_config = read_rags_config()
    rag_name = args.name
    if not rag_name:
        questions = [
            inquirer.List('rag_name',
                          message="Which RAG index do you want to ingest data into?",
                          choices=[rag['name'] for rag in rags_config],
                          ),
        ]
        answers = inquirer.prompt(questions)
        rag_name = answers['rag_name']

    selected_rag = next((rag for rag in rags_config if rag['name'] == rag_name), None)
    if not selected_rag:
        print(f"Error: RAG index '{rag_name}' not found.")
        return

    print(f"Ingesting data into RAG index '{rag_name}'...")
    
    qdrant_client = QdrantClient(path=VECTOR_STORAGE_PATH)
    
    ingestion_toolkit = IngestionToolkit()
    indexing_toolkit = IndexingToolkit(qdrant_client, selected_rag['collection_name'])
    
    ingestion_agent = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )
    
    workforce = Workforce(
        "Ingestion and Indexing Workforce",
        coordinator_agent=ingestion_agent,
    )
    
    workforce.add_single_agent_worker(
        "Data Ingestion Specialist",
        worker=ingestion_agent,
        tools=ingestion_toolkit.get_tools(),
    )
    workforce.add_single_agent_worker(
        "Data Indexing Specialist",
        worker=ingestion_agent,
        tools=indexing_toolkit.get_tools(),
    )
    
    task = Task(
        f"Ingest and index documents from {args.path} into the collection {selected_rag['collection_name']}"
    )
    result = workforce.process_task(task)
    print(result)
    print("Data ingestion completed.")

def ask_question(args):
    """Asks a question to a specified RAG index."""
    rags_config = read_rags_config()
    rag_name = args.name
    if not rag_name:
        questions = [
            inquirer.List('rag_name',
                          message="Which RAG index do you want to ask a question to?",
                          choices=[rag['name'] for rag in rags_config],
                          ),
        ]
        answers = inquirer.prompt(questions)
        rag_name = answers['rag_name']

    selected_rag = next((rag for rag in rags_config if rag['name'] == rag_name), None)
    if not selected_rag:
        print(f"Error: RAG index '{rag_name}' not found.")
        return

    print(f"Asking question to RAG index '{rag_name}'...")

    tool_critic_model = ModelFactory.create(
        model_platform=ModelPlatformType.MISTRAL,
        model_type="open-mistral-7b",
    )
    retrieval_model = ModelFactory.create(
        model_platform=ModelPlatformType.MISTRAL,
        model_type="open-mistral-7b",
    )
    synthesis_model = ModelFactory.create(
        model_platform=ModelPlatformType.MISTRAL,
        model_type=ModelType.MISTRAL_LARGE,
    )

    tool_critic_agent = ToolCriticAgent(model=tool_critic_model)
    retrieval_agent = RetrievalAgent(model=retrieval_model)
    synthesis_agent = SynthesisAgent(model=synthesis_model)

    workforce = Workforce(
        "RAG Query Workforce",
        coordinator_agent=tool_critic_agent,
    )
    workforce.add_single_agent_worker(
        "Retrieval Specialist",
        worker=retrieval_agent,
    )
    workforce.add_single_agent_worker(
        "Synthesis Specialist",
        worker=synthesis_agent,
    )

    task = Task(f"Answer the following question: {args.query}")
    result = workforce.process_task(task)
    
    print("Answer:")
    print(result)

def main():
    parser = argparse.ArgumentParser(
        description="A CLI tool for managing and querying multiple RAG indexes."
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    create_parser = subparsers.add_parser("create", help="Create a new RAG index.")
    create_parser.add_argument("name", help="The name of the RAG index to create.")
    create_parser.set_defaults(func=create_rag)
    
    list_parser = subparsers.add_parser("list", help="List all available RAG indexes.")
    list_parser.set_defaults(func=list_rags)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest data into a RAG index.")
    ingest_parser.add_argument("path", help="Path to a file, directory, or Git repository URL to process.")
    ingest_parser.add_argument("--name", help="The name of the RAG index to ingest data into.")
    ingest_parser.set_defaults(func=ingest_data)
    
    ask_parser = subparsers.add_parser("ask", help="Ask a question to a RAG index.")
    ask_parser.add_argument("query", help="The question to ask.")
    ask_parser.add_argument("--name", help="The name of the RAG index to ask the question to.")
    ask_parser.set_defaults(func=ask_question)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()