# Document Summarizer and RAG Pipeline

This project implements a sophisticated, multi-agent Retrieval-Augmented Generation (RAG) pipeline for document summarization and question-answering. It is built using the **CAMEL-AI** framework, leveraging a `Workforce` of specialized agents that use **Tools** to handle complex workflows in a modular and scalable way.

## Architecture: A Hierarchical Team of Agents

The pipeline is designed as a hierarchical, multi-agent system that optimizes for cost, latency, and quality. It is built using the **CAMEL-AI** framework and features a team of specialized agents, each with a distinct role and a detailed system prompt defined in `rag_builder/prompts.py`.

### The Three-Tiered Agent Structure

1.  **The Analyst (ClarificationAgent)**:
    -   **Model**: `openai/gpt-oss-20b:free` (via OpenRouter)
    -   **Objective**: To perform a deep analysis of the user's query. It classifies intent, segments complex questions, extracts key entities, and formulates a refined, context-aware query for the retrieval system. This pre-retrieval step ensures that the search is highly targeted and relevant.

2.  **The Retriever (HybridRetriever)**:
    -   **Technology**: Qdrant, SentenceTransformers, BM25
    -   **Objective**: To execute a broad, two-pronged search. It uses the `refined_query_for_embedding` from the Analyst for a semantic vector search and the `search_terms` for a parallel keyword (BM25) search. It fetches a large set of candidate documents and then reranks them using a hybrid score to find the most relevant context.

3.  **The Executive (SynthesisAgent)**:
    -   **Model**: `openai/gpt-oss-20b:free` (via OpenRouter)
    -   **Objective**: To synthesize the user query and the context package prepared by the Retriever into a final, high-quality, and grounded answer. It is mandated to cite sources, rely exclusively on the provided context for its primary answer, and adhere to strict formatting rules.

### Data Ingestion and Indexing

Before queries can be answered, documents are processed by a separate `Workforce` of agents dedicated to ingestion and indexing:

-   **Universal and Recursive Data Loading**: A versatile `load_from_path` tool intelligently handles Git repositories, local directories, and single files to fetch all source materials.
-   **Indexing Agent**: This agent uses an `IndexingToolkit` to:
    -   **`chunk_document`**: Perform structure-aware chunking using **AST parsing** for code and a hierarchical approach for Markdown. For Jupyter Notebooks, each cell is treated as a chunk.
    -   **`embed_chunks`**: Convert text chunks into vectors using the `all-MiniLM-L6-v2` model from **SentenceTransformers**.
    -   **`deduplicate_and_store`**: Ensure unique content is stored in the **Qdrant** vector database.

## Features

-   **Advanced Query Analysis**: A dedicated `ClarificationAgent` performs intent classification, entity extraction, and query refinement to ensure a deep understanding of the user's needs before retrieval.
-   **Hybrid Search**: Combines semantic (vector) search with traditional keyword (BM25) search to improve retrieval accuracy for a wide range of queries.
-   **Hierarchical Agent Workflow**: A cost and latency-optimized three-tiered system (Analyst, Retriever, Executive) ensures that the right level of intelligence is applied at each stage of the query process.
-   **Universal and Recursive Data Ingestion**: A single `load_from_path` tool intelligently handles Git repositories, local directories, and individual files.
-   **Local Embeddings**: Uses **SentenceTransformers** with the `all-MiniLM-L6-v2` model to generate embeddings locally, ensuring data privacy and removing reliance on external APIs.
-   **Advanced Structure-Aware Chunking**: Employs **AST parsing** for code and hierarchical analysis for Markdown to create context-rich chunks.
-   **High-Quality Synthesis**: Leverages the `openai/gpt-oss-20b:free` model as the Executive agent to provide polished, definitive answers.
-   **Efficient Storage**: Uses **Qdrant** for high-performance vector search and **embedding deduplication** to avoid redundant data.

## Setup and Usage

### 1. Installation

First, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd doc-summarizer
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root of the project and add your OpenRouter API key.

```
OPENROUTER_API_KEY="your-openrouter-api-key"
# Add any other required keys (e.g., for data loaders)
```

### 3. Running the Pipeline

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

Once your data is ingested, you can ask questions. The pipeline will retrieve the relevant context and generate an answer.

```bash
python -m rag_builder.main ask "What is the Workforce module in CAMEL AI?" --name my-document-index
```
