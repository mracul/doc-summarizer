# Document Summarizer and RAG Pipeline

This project implements a sophisticated, multi-agent Retrieval-Augmented Generation (RAG) pipeline for document summarization and question-answering. It is built using the **CAMEL-AI** framework, leveraging a `Workforce` of specialized agents that use **Tools** to handle complex workflows in a modular and scalable way.

## Architecture: A Hierarchical Team of Agents

The pipeline is designed as a hierarchical, multi-agent system that optimizes for cost, latency, and quality. It is built using the **CAMEL-AI** framework and features a team of specialized agents, each with a distinct role and a detailed system prompt defined in `rag_builder/prompts.py`.

### The Three-Tiered Agent Structure

1.  **The Front Desk (ToolCriticAgent)**:
    -   **Model**: `open-mistral-7b` (or a similar low-latency model)
    -   **Objective**: To function as a high-accuracy query router. It analyzes an incoming user query and selects the single most appropriate tool for the initial data retrieval step, saving cost and time.

2.  **The Specialist (RetrievalAgent)**:
    -   **Model**: `open-mistral-7b` (or a similar capable reasoning model)
    -   **Objective**: To execute a dynamic, multi-step retrieval process. When a query requires deep investigation, this agent performs contextual analysis, query expansion (multi-hop retrieval), and final curation to assemble a comprehensive and highly relevant context package.

3.  **The Executive (SynthesisAgent)**:
    -   **Model**: `mistral-large-latest`
    -   **Objective**: To synthesize the user query and the context package prepared by the Specialist into a final, high-quality, and grounded answer. It is mandated to cite sources, rely exclusively on the provided context for its primary answer, and adhere to strict formatting rules.

### Data Ingestion and Indexing

Before queries can be answered, documents are processed by a separate `Workforce` of agents dedicated to ingestion and indexing:

-   **Universal and Recursive Data Loading**: A versatile `load_from_path` tool intelligently handles Git repositories, local directories, and single files to fetch all source materials.
-   **Indexing Agent**: This agent uses an `IndexingToolkit` to:
    -   **`chunk_document`**: Perform structure-aware chunking using **AST parsing** for code and a hierarchical approach for Markdown. For Jupyter Notebooks, each cell is treated as a chunk.
    -   **`embed_chunks`**: Convert text chunks into vectors using a local **Ollama** instance running the `mistral` model.
    -   **`deduplicate_and_store`**: Ensure unique content is stored in the **Qdrant** vector database.

## Features

-   **Hierarchical Agent Workflow**: A cost and latency-optimized three-tiered system (Front Desk, Specialist, Executive) ensures that the right level of intelligence is applied at each stage of the query process.
-   **Universal and Recursive Data Ingestion**: A single `load_from_path` tool intelligently handles Git repositories, local directories, and individual files.
-   **Local Embeddings**: Uses **Ollama** with the `mistral` model to generate embeddings locally, ensuring data privacy and removing reliance on external APIs.
-   **Advanced Structure-Aware Chunking**: Employs **AST parsing** for code and hierarchical analysis for Markdown to create context-rich chunks.
-   **Multi-Hop Reasoning for Retrieval**: The Specialist agent performs deep contextual analysis and query expansion to build a comprehensive, noise-free context.
-   **High-Quality Synthesis**: Leverages **Mistral Large** as the Executive agent to provide polished, definitive answers.
-   **Efficient Storage**: Uses **Qdrant** for high-performance vector search and **embedding deduplication** to avoid redundant data.

## Setup and Usage

### 1. Prerequisites

Before running the pipeline, you need to have **Ollama** installed and running.

-   **Install Ollama**: Follow the instructions at [https://ollama.com/](https://ollama.com/).
-   **Pull the Mistral Model**: Once Ollama is running, pull the `mistral` model, which will be used for generating embeddings.
    ```bash
    ollama pull mistral
    ```

### 2. Installation

First, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd doc-summarizer
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root of the project and add your API keys. You will need a key for Mistral AI for the final synthesis step.

```
MISTRAL_API_KEY="your-mistral-api-key"
# Add any other required keys (e.g., for data loaders)
```

### 4. Running the Pipeline

The pipeline is managed through the main CLI script.

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

Once your data is ingested, you can ask questions. The pipeline will retrieve the relevant context and generate an answer using Mistral Large.

```bash
python -m rag_builder.main ask "What is the Workforce module in CAMEL AI?" --name my-document-index
```
