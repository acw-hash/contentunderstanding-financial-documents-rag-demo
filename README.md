# Content Understanding Financial Documents RAG Demo

A RAG (Retrieval-Augmented Generation) demo that uses **Azure AI Content Understanding** to analyze financial documents (PDFs), index their content into **Azure AI Search**, and answer questions via an interactive chat powered by **Azure OpenAI**.

## How it works

1. **Analyze** – `cu_pipeline.py` submits financial documents to an Azure AI Content Understanding analyzer, which extracts narrative text, tables, and figures.
2. **Chunk & embed** – The extracted content is split into typed chunks (narrative / table / figure) and embedded using Azure OpenAI embeddings.
3. **Index** – Chunks are indexed in an Azure AI Search vector store.
4. **Chat** – `rag_chat.py` runs an interactive loop that retrieves the most relevant chunks and answers questions using an Azure OpenAI chat model.

## Prerequisites

- Python 3.9+
- An Azure subscription with the following resources provisioned:
  - Azure AI Services (Content Understanding)
  - Azure OpenAI (a chat deployment and an embedding deployment)
  - Azure AI Search

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**

   Copy the variables below into a `.env` file in the project root and fill in your values:

   ```env
   # Azure AI Content Understanding
   AZURE_AI_SERVICE_ENDPOINT=https://<your-ai-service>.cognitiveservices.azure.com/

   # Azure OpenAI
   AZURE_OPENAI_ENDPOINT=https://<your-openai>.openai.azure.com/
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=<chat-deployment>
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=<embedding-deployment>

   # Azure AI Search
   AZURE_SEARCH_ENDPOINT=https://<your-search>.search.windows.net
   # AZURE_SEARCH_ADMIN_KEY=<key>   # Optional; omit to use Entra ID (DefaultAzureCredential)
   # AZURE_SEARCH_INDEX_NAME=sample-doc-index   # Optional; default shown

   # Content Understanding model deployments (optional overrides)
   # CU_COMPLETION_MODEL_NAME=gpt-4.1
   # CU_EMBEDDING_MODEL_NAME=text-embedding-3-small
   # CU_DEFAULT_COMPLETION_DEPLOYMENT=<chat-deployment>
   # CU_DEFAULT_EMBEDDING_DEPLOYMENT=<embedding-deployment>

   # Paths (optional overrides)
   # CU_DATA_DIR=./data
   # CU_TEMPLATE_DIR=./analyzer_templates
   ```

3. **Place your financial document**

   Put your PDF in the `data/` directory and update the `location` field in `ANALYZER_CONFIGS` inside `cu_pipeline.py` to point to it (default: `data/sample_report.pdf`).

## Usage

### First run — create analyzers and start chat

```bash
python cu_pipeline.py --create-analyzer
```

This registers the Content Understanding analyzer templates, runs analysis, indexes the document, and launches the interactive chat.

### Subsequent runs

```bash
python cu_pipeline.py
```

Analyzers are reused across runs; no need to recreate them unless templates change.

### Pipeline only (no chat)

```bash
python cu_pipeline.py --skip-chat
```

### Chat only (reuse an already-indexed vector store)

```bash
python rag_chat.py
```

## Project structure

```
cu_pipeline.py          # Ingestion pipeline (analyze → chunk → embed → index)
rag_chat.py             # Interactive RAG chat loop
analyzer_templates/     # Content Understanding analyzer JSON templates
data/                   # Input documents (PDF)
requirements.txt
```

## Authentication

The demo uses `DefaultAzureCredential` (Entra ID) by default. To use API key authentication for Azure AI Search instead, set `AZURE_SEARCH_ADMIN_KEY` in `.env`.
