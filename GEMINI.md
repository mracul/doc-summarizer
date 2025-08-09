# Project Overview

This project is a sophisticated, multi-agent Retrieval-Augmented Generation (RAG) pipeline for document summarization and question-answering. It is built using the **CAMEL-AI** framework, leveraging a `Workforce` of specialized agents that use **Tools** to handle complex workflows in a modular and scalable way.

The pipeline is designed as a hierarchical, multi-agent system that optimizes for cost, latency, and quality. It features a team of specialized agents, each with a distinct role and a detailed system prompt defined in `rag_builder/prompts.py`.

The core technologies used are:
- **CAMEL-AI**: For the multi-agent framework.
- **Qdrant**: For the vector database.
- **SentenceTransformers**: For local embeddings.
- **OpenRouter**: For accessing various language models.
- **Unstructured**: For document parsing and chunking.
- **Prompt-toolkit**: For the interactive TUI.

# Building and Running

## 1. Installation

First, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd doc-summarizer
pip install -r requirements.txt
```

## 2. Configuration

Create a `.env` file in the root of the project and add your OpenRouter API key.

```
OPENROUTER_API_KEY="your-openrouter-api-key"
# Add any other required keys (e.g., for data loaders)
```

## 3. Running the Pipeline

The pipeline can be run in two modes:

### Interactive TUI

To launch the interactive Text-based User Interface (TUI), run:

```bash
python -m rag_builder.main
```

The TUI provides a user-friendly interface for managing RAG indexes, ingesting data, and asking questions.

### Command-Line Interface (CLI)

The pipeline can also be controlled via CLI commands:

**A. Create a RAG Index**

First, create a named index where your data will be stored.

```bash
python -m rag_builder.main create my-document-index
```

**B. Ingest Data**

Next, ingest your documents into the index. You can point to a local file, a directory, or a Git repository URL.

```bash
# Ingest a local directory
python -m rag_builder.main ingest ./path/to/your/docs --name my-document-index

# Ingest a Git repository
python -m rag_builder.main ingest https://github.com/camel-ai/camel.git --name my-document-index
```

**C. Ask a Question**

Once your data is ingested, you can ask questions. The pipeline will retrieve the relevant context and generate an answer.

```bash
python -m rag_builder.main ask "What is the Workforce module in CAMEL AI?" --name my-document-index
```

# Development Conventions

- **Agent-based Architecture**: The core logic is encapsulated in specialized agents, each with a specific role. New functionality should be implemented by creating new agents or extending existing ones.
- **Toolkit-based Functionality**: Agents are equipped with toolkits that provide specific functionalities. New tools should be added to the appropriate toolkit or a new toolkit should be created.
- **Configuration Management**: The project uses a `rags.json` file to manage RAG indexes and their corresponding Qdrant collections.
- **Asynchronous Operations**: The project uses `asyncio` to handle asynchronous operations, particularly for the TUI and model interactions.
- **Dependency Management**: Project dependencies are managed in the `requirements.txt` file.
