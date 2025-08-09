import argparse
import uuid
import inquirer
from dotenv import load_dotenv
import sys
import asyncio

from rag_builder.pipeline import RAGPipeline
from rag_builder.utils import read_rags_config, write_rags_config

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

    pipeline = RAGPipeline()
    pipeline.ingest(args.path, selected_rag['collection_name'])

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

    pipeline = RAGPipeline()
    pipeline.ask(args.query, selected_rag['collection_name'])

def main():
    """Main function to parse arguments and call the appropriate command."""
    if len(sys.argv) == 1:
        # No arguments provided, run the TUI
        from rag_builder import tui
        asyncio.run(tui.main())
        return

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
