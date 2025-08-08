import argparse
import os
import shutil
import tempfile
import uuid
from dotenv import load_dotenv
from git import Repo
import inquirer

from camel.loaders import UnstructuredIO, MistralReader
from camel.embeddings import MistralEmbedding
from camel.storages import QdrantStorage
from camel.retrievers import AutoRetriever
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory

from rag_builder.config import (
    VECTOR_STORAGE_PATH,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    OCR_MODEL,
)
from rag_builder.utils import read_rags_config, write_rags_config

# Load environment variables from .env file
load_dotenv()

def is_git_url(path: str) -> bool:
    """Checks if the given path is a valid Git repository URL."""
    return path.startswith(("http://", "https://")) and path.endswith(".git")

def create_rag(args):
    """Creates a new RAG index configuration."""
    print(f"Creating a new RAG index named '{args.name}'...")
    rags_config = read_rags_config()
    
    # Check if a RAG with the same name already exists
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
    if not rags_config:
        print("No RAG indexes found. Please create one first.")
        return

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

    embedding_model = MistralEmbedding(model_name=EMBEDDING_MODEL)
    vector_storage = QdrantStorage(
        vector_dim=embedding_model.get_output_dim(),
        path=VECTOR_STORAGE_PATH,
        collection_name=selected_rag['collection_name'],
    )
    retriever = AutoRetriever(
        vector_storage=vector_storage,
        embedding_model=embedding_model,
    )

    files_to_process = []
    temp_dir = None

    if is_git_url(args.path):
        try:
            temp_dir = tempfile.mkdtemp()
            print(f"Cloning repository from {args.path} to {temp_dir}...")
            Repo.clone_from(args.path, temp_dir)
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    files_to_process.append(os.path.join(root, file))
        except Exception as e:
            print(f"Error cloning repository: {e}")
            if temp_dir:
                shutil.rmtree(temp_dir)
            return
    elif os.path.isfile(args.path):
        files_to_process = [args.path]
    elif os.path.isdir(args.path):
        for root, _, files in os.walk(args.path):
            for file in files:
                files_to_process.append(os.path.join(root, file))
    else:
        print(f"Error: The path '{args.path}' is not a valid file, directory, or Git URL.")
        return

    accepted_file_types = [
    ".py", ".ipynb", ".md", ".mdx", ".rst", ".txt", ".json", 
    ".yaml", ".yml", ".js", ".ts", ".java", ".cpp", ".c", 
    ".go", ".rb", ".sh", ".html", ".xml", ".csv", ".tsv"
    ]
    
    filtered_files = [
        f for f in files_to_process 
        if any(f.endswith(ext) for ext in accepted_file_types)
    ]

    for file_path in filtered_files:
        print(f"Processing file: {file_path}")
        try:
            if args.ocr and file_path.lower().endswith(".pdf"):
                print("Using OCR to process PDF...")
                mistral_reader = MistralReader(model=OCR_MODEL)
                documents = mistral_reader.load_data(file_path)
                retriever.process(documents)
            else:
                loader = UnstructuredIO()
                documents = loader.load_data(file_path)
                retriever.process(documents)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if temp_dir:
        shutil.rmtree(temp_dir)

    print("Data ingestion completed.")

def ask_question(args):
    """Asks a question to a specified RAG index."""
    rags_config = read_rags_config()
    if not rags_config:
        print("No RAG indexes found. Please create one first.")
        return

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

    embedding_model = MistralEmbedding(model_name=EMBEDDING_MODEL)
    vector_storage = QdrantStorage(
        vector_dim=embedding_model.get_output_dim(),
        path=VECTOR_STORAGE_PATH,
        collection_name=selected_rag['collection_name'],
    )
    retriever = AutoRetriever(
        vector_storage=vector_storage,
        embedding_model=embedding_model,
    )

    retrieved_info = retriever.run_vector_retriever(
        query=args.query,
    )
    
    chat_model = ModelFactory.create(
        model_platform=ModelPlatformType.MISTRAL,
        model_type=CHAT_MODEL,
    )

    agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Assistant",
            content="You are a helpful assistant that answers questions based on the provided context."
        ),
        model=chat_model,
    )
    
    context = "\n".join([info['text'] for info in retrieved_info['Retrieved Context']])
    prompt = f"Based on the following context, please answer the question: {args.query}\n\nContext:\n{context}"
    
    user_message = BaseMessage.make_user_message(role_name="User", content=prompt)
    response = agent.step(user_message)
    
    print("Answer:")
    print(response.msg.content)

def main():
    """Main function to parse arguments and call the appropriate command."""
    parser = argparse.ArgumentParser(
        description="A CLI tool for managing and querying multiple RAG indexes."
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    # Parser for the "create" command
    create_parser = subparsers.add_parser("create", help="Create a new RAG index.")
    create_parser.add_argument("name", help="The name of the RAG index to create.")
    create_parser.set_defaults(func=create_rag)
    
    # Parser for the "list" command
    list_parser = subparsers.add_parser("list", help="List all available RAG indexes.")
    list_parser.set_defaults(func=list_rags)

    # Parser for the "ingest" command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data into a RAG index.")
    ingest_parser.add_argument("path", help="Path to a file, directory, or Git repository URL to process.")
    ingest_parser.add_argument("--name", help="The name of the RAG index to ingest data into.")
    ingest_parser.add_argument("--ocr", action="store_true", help="Enable OCR for PDF files.")
    ingest_parser.set_defaults(func=ingest_data)
    
    # Parser for the "ask" command
    ask_parser = subparsers.add_parser("ask", help="Ask a question to a RAG index.")
    ask_parser.add_argument("query", help="The question to ask.")
    ask_parser.add_argument("--name", help="The name of the RAG index to ask the question to.")
    ask_parser.set_defaults(func=ask_question)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
