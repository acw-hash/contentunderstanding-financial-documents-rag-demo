import os
import time
import argparse
from pathlib import Path

from pypdf import PdfReader
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_API_VERSION = os.getenv("AZURE_OPENAI_CHAT_API_VERSION") or "2024-08-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION") or "2023-05-15"
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
BASE_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME") or "sample-doc-index"
AZURE_SEARCH_PIPELINE_INDEX_NAME = os.getenv("AZURE_SEARCH_PIPELINE_INDEX_NAME") or f"{BASE_INDEX_NAME}-search-only"
DATA_DIR = Path(os.getenv("CU_DATA_DIR", str(SCRIPT_DIR / "data")))

# Tuned for long, dense 10-K style filings.
NARRATIVE_CHUNK_SIZE = int(os.getenv("SEARCH_PIPELINE_CHUNK_SIZE", "2200"))
NARRATIVE_CHUNK_OVERLAP = int(os.getenv("SEARCH_PIPELINE_CHUNK_OVERLAP", "300"))
RETRIEVAL_K = int(os.getenv("SEARCH_PIPELINE_TOP_K", "5"))

_PROMPT_STR = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")


# ---------------------------------------------------------------------------
# Ingestion and chunking
# ---------------------------------------------------------------------------
def _resolve_path(path_like: Path | str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def _extract_pdf_pages(pdf_path: Path) -> list[Document]:
    docs = []
    reader = PdfReader(str(pdf_path))
    for page_idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page": page_idx + 1,
                },
            )
        )
    return docs


def _discover_pdfs(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.pdf"))
    if not files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. Place one or more 10-K PDFs there or set CU_DATA_DIR."
        )
    return files


def load_and_chunk_pdfs(data_dir: Path) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=NARRATIVE_CHUNK_SIZE,
        chunk_overlap=NARRATIVE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs = []
    pdf_paths = _discover_pdfs(data_dir)
    print(f"Found {len(pdf_paths)} PDF file(s) in {data_dir}.")

    for pdf_path in pdf_paths:
        page_docs = _extract_pdf_pages(pdf_path)
        if not page_docs:
            print(f"Skipping {pdf_path.name}: no extractable text found.")
            continue
        chunked = splitter.split_documents(page_docs)
        for idx, doc in enumerate(chunked):
            doc.metadata["chunk_index"] = idx
            doc.metadata["chunk_strategy"] = "search_only_10k_v1"
        all_docs.extend(chunked)
        print(f"Chunked {pdf_path.name}: {len(chunked)} chunk(s).")

    if not all_docs:
        raise RuntimeError("No chunks produced from PDFs. Check PDF text extractability.")

    print(f"Prepared {len(all_docs)} total chunk(s) for indexing.")
    return all_docs


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def embed_and_index_chunks(docs: list[Document]) -> AzureSearch:
    aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
    )

    vector_store = AzureSearch.from_documents(
        documents=docs,
        embedding=aoai_embeddings,
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_ADMIN_KEY,
        index_name=AZURE_SEARCH_PIPELINE_INDEX_NAME,
    )
    print(f"Indexed {len(docs)} chunks into '{AZURE_SEARCH_PIPELINE_INDEX_NAME}'.")
    return vector_store


# ---------------------------------------------------------------------------
# Retrieval and chat
# ---------------------------------------------------------------------------
def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_CHAT_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        azure_ad_token_provider=token_provider,
        temperature=0.7,
    )


def retrieve_with_scores(vector_store: AzureSearch, query: str, k: int = RETRIEVAL_K):
    try:
        results = vector_store.similarity_search_with_relevance_scores(query, k=k, search_type="hybrid")
    except Exception:
        docs = vector_store.similarity_search(query, k=k, search_type="hybrid")
        results = [(doc, None) for doc in docs]

    print("\nRetrieved chunks:")
    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata or {}
        source = meta.get("source_file", "unknown")
        page = meta.get("page", "?")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        preview = " ".join(doc.page_content.split())[:280]
        print(f"[{i}] source={source} page={page} score={score_text}")
        print(f"    {preview}")

    return results


def answer_query(llm: AzureChatOpenAI, query: str, retrieved):
    context = "\n\n".join(doc.page_content for doc, _ in retrieved)
    prompt = ChatPromptTemplate.from_template(_PROMPT_STR)
    messages = prompt.format_messages(question=query, context=context)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def run_chat(vector_store: AzureSearch, startup_started_at=None):
    llm = _build_llm()
    print("Azure AI Search chat ready. Press Enter with no input to quit.")
    startup_timer_logged = False

    while True:
        if not startup_timer_logged and startup_started_at is not None:
            startup_elapsed = time.perf_counter() - startup_started_at
            print(f"[timer] Startup to first query prompt: {startup_elapsed:.2f}s")
            startup_timer_logged = True

        query = input("Enter your query: ")
        if query == "":
            break

        query_started_at = time.perf_counter()
        retrieved = retrieve_with_scores(vector_store, query, k=RETRIEVAL_K)
        answer = answer_query(llm, query, retrieved)
        print("\nAnswer:")
        print(answer)
        query_elapsed = time.perf_counter() - query_started_at
        print(f"[timer] Query to answer returned: {query_elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(data_dir: Path | str = DATA_DIR) -> AzureSearch:
    resolved_data_dir = _resolve_path(data_dir)
    docs = load_and_chunk_pdfs(resolved_data_dir)
    vector_store = embed_and_index_chunks(docs)
    return vector_store


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    script_started_at = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Azure AI Search-only RAG pipeline for PDF filings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Run pipeline + chat:   python azure-ai-search-pipeline.py\n"
            "  Pipeline only:         python azure-ai-search-pipeline.py --skip-chat\n"
            "  Custom data folder:    python azure-ai-search-pipeline.py --data-dir ./data\n"
        ),
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        default=False,
        help="Run the pipeline only without launching the interactive chat loop.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory containing PDF files to ingest (defaults to CU_DATA_DIR or ./data).",
    )
    args = parser.parse_args()

    _vector_store = run_pipeline(data_dir=args.data_dir)

    if not args.skip_chat:
        run_chat(_vector_store, startup_started_at=script_started_at)
