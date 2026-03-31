import os
import json
import argparse
import time
import re
from pathlib import Path

from dotenv import load_dotenv
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.ai.contentunderstanding import ContentUnderstandingClient

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AZURE_AI_SERVICE_ENDPOINT = os.getenv("AZURE_AI_SERVICE_ENDPOINT")
AZURE_AI_SERVICE_API_VERSION = os.getenv("AZURE_AI_SERVICE_API_VERSION") or "2024-12-01-preview"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION") or "2023-05-15"
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME") or "sample-doc-index"
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
CU_COMPLETION_MODEL_NAME = os.getenv("CU_COMPLETION_MODEL_NAME") or "gpt-4.1"
CU_EMBEDDING_MODEL_NAME = os.getenv("CU_EMBEDDING_MODEL_NAME") or "text-embedding-3-small"
CU_DEFAULT_COMPLETION_DEPLOYMENT = os.getenv("CU_DEFAULT_COMPLETION_DEPLOYMENT") or AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
CU_DEFAULT_EMBEDDING_DEPLOYMENT = os.getenv("CU_DEFAULT_EMBEDDING_DEPLOYMENT") or AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
DATA_DIR = Path(os.getenv("CU_DATA_DIR", str(SCRIPT_DIR / "data")))
TEMPLATE_DIR = Path(os.getenv("CU_TEMPLATE_DIR", str(SCRIPT_DIR / "analyzer_templates")))

# Chunking defaults tuned for long financial reports (10-K/10-Q/annual reports).
NARRATIVE_CHUNK_SIZE = int(os.getenv("NARRATIVE_CHUNK_SIZE", "2200"))
NARRATIVE_CHUNK_OVERLAP = int(os.getenv("NARRATIVE_CHUNK_OVERLAP", "300"))
TABLE_CHUNK_SIZE = int(os.getenv("TABLE_CHUNK_SIZE", "3200"))
TABLE_CHUNK_OVERLAP = int(os.getenv("TABLE_CHUNK_OVERLAP", "150"))
FIGURE_CHUNK_SIZE = int(os.getenv("FIGURE_CHUNK_SIZE", "1800"))
FIGURE_CHUNK_OVERLAP = int(os.getenv("FIGURE_CHUNK_OVERLAP", "200"))

# Shared credential and token provider (imported by rag_chat.py)
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

# ---------------------------------------------------------------------------
# Analyzer configuration
# Stable IDs (no UUID) so the same analyzers are reused across runs.
# ---------------------------------------------------------------------------
ANALYZER_CONFIGS = [
    {
        "id": "cu_doc_analyzer",
        "template_path": TEMPLATE_DIR / "content_document.json",
        "location": DATA_DIR / "sample_report.pdf",
    },
    # {
    #     "id": "cu_doc_layout_analyzer",
    #     "template_path": TEMPLATE_DIR / "content_document_layout.json",
    #     "location": DATA_DIR / "sample_report.pdf",
    # },
    {
        "id": "cu_image_analyzer",
        "template_path": TEMPLATE_DIR / "image_chart.json",
        "location": DATA_DIR / "sample_report.pdf",
    },
    # {
    #     "id": "cu_audio_analyzer",
    #     "template_path": TEMPLATE_DIR / "call_recording_analytics.json",
    #     "location": DATA_DIR / "callCenterRecording.mp3",
    # },
    # {
    #     "id": "cu_video_analyzer",
    #     "template_path": TEMPLATE_DIR / "video_content_understanding.json",
    #     "location": DATA_DIR / "FlightSimulator.mp4",
    # },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_cu_client() -> ContentUnderstandingClient:
    return ContentUnderstandingClient(
        endpoint=AZURE_AI_SERVICE_ENDPOINT,
        credential=credential,
    )


def _convert_values_to_strings(json_obj):
    return [str(value) for value in json_obj]


def _to_json_text(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _extract_page_hint(text: str):
    match = re.search(r"\bpage\s*[:#-]?\s*(\d{1,4})\b", text, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _classify_content_type(analyzer_id: str, text: str) -> str:
    if analyzer_id == "cu_image_analyzer":
        return "figure"

    lower = text.lower()
    table_markers = ["|", "markdown table", "row", "column", "cell", "\tt"]
    figure_markers = ["chart", "graph", "figure", "axis", "legend", "trend"]

    table_score = sum(marker in lower for marker in table_markers)
    figure_score = sum(marker in lower for marker in figure_markers)

    if table_score >= 2:
        return "table"
    if figure_score >= 2:
        return "figure"
    return "narrative"


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_splitter_for_content_type(content_type: str) -> RecursiveCharacterTextSplitter:
    if content_type == "table":
        return RecursiveCharacterTextSplitter(
            chunk_size=TABLE_CHUNK_SIZE,
            chunk_overlap=TABLE_CHUNK_OVERLAP,
            separators=["\n|", "\n\n", "\n", " ", ""],
        )
    if content_type == "figure":
        return RecursiveCharacterTextSplitter(
            chunk_size=FIGURE_CHUNK_SIZE,
            chunk_overlap=FIGURE_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=NARRATIVE_CHUNK_SIZE,
        chunk_overlap=NARRATIVE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _resolve_path(path_like: Path | str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def _prepare_analyzer_resource(analyzer_resource: dict) -> dict:
    resource = dict(analyzer_resource)
    base_analyzer_map = {
        "prebuilt-documentAnalyzer": "prebuilt-document",
        "prebuilt-imageAnalyzer": "prebuilt-image",
        "prebuilt-audioAnalyzer": "prebuilt-audio",
        "prebuilt-videoAnalyzer": "prebuilt-video",
    }
    base_analyzer_id = resource.get("baseAnalyzerId")
    if base_analyzer_id in base_analyzer_map:
        resource["baseAnalyzerId"] = base_analyzer_map[base_analyzer_id]

    # Image analyzer doesn't support models configuration
    if base_analyzer_id not in ("prebuilt-imageAnalyzer", "prebuilt-image"):
        models = dict(resource.get("models") or {})
        models.setdefault("completion", CU_COMPLETION_MODEL_NAME)
        models.setdefault("embedding", CU_EMBEDDING_MODEL_NAME)
        resource["models"] = models
    return resource


def _ensure_cu_defaults(client: ContentUnderstandingClient) -> None:
    try:
        defaults = client.get_defaults()
        if defaults and getattr(defaults, "model_deployments", None):
            return
    except Exception:
        pass

    if not CU_DEFAULT_COMPLETION_DEPLOYMENT or not CU_DEFAULT_EMBEDDING_DEPLOYMENT:
        raise RuntimeError(
            "Content Understanding defaults are not set. Configure resource defaults in Studio, or set "
            "CU_DEFAULT_COMPLETION_DEPLOYMENT and CU_DEFAULT_EMBEDDING_DEPLOYMENT in .env."
        )

    try:
        client.update_defaults(
            model_deployments={
                CU_COMPLETION_MODEL_NAME: CU_DEFAULT_COMPLETION_DEPLOYMENT,
                CU_EMBEDDING_MODEL_NAME: CU_DEFAULT_EMBEDDING_DEPLOYMENT,
            }
        )
    except HttpResponseError as e:
        error_str = str(e)
        if "DeploymentIdNotSupported" in error_str:
            raise RuntimeError(
                "Content Understanding can't use the configured completion deployment. "
                f"Current completion deployment: '{CU_DEFAULT_COMPLETION_DEPLOYMENT}'. "
                "Deploy a supported completion model and point CU defaults to it. "
                "Supported completion models include: gpt-4o, gpt-4o-mini, gpt-4.1, "
                "gpt-4.1-mini, gpt-4.1-nano, and gpt-5.2. "
                "Then set CU_COMPLETION_MODEL_NAME and CU_DEFAULT_COMPLETION_DEPLOYMENT in .env."
            ) from e
        raise RuntimeError(
            "Failed to configure Content Understanding defaults automatically. "
            f"Service error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def create_analyzers():
    """Create Content Understanding analyzers (one-time setup).

    Only needs to be run once, or again when analyzer templates change.
    Pass --create-analyzer on the command line to trigger this step.
    """
    client = _get_cu_client()
    _ensure_cu_defaults(client)
    for analyzer in ANALYZER_CONFIGS:
        analyzer_id = analyzer["id"]
        template_path = analyzer["template_path"]
        try:
            template_full_path = _resolve_path(template_path)
            if not template_full_path.exists():
                print(
                    f"Skipping analyzer '{analyzer_id}': template file not found at {template_full_path}."
                )
                continue
            with template_full_path.open("r", encoding="utf-8") as f:
                analyzer_resource = _prepare_analyzer_resource(json.load(f))
            response = client.begin_create_analyzer(
                analyzer_id=analyzer_id,
                resource=analyzer_resource,
            )
            response.result()
            print(f"Successfully created analyzer: {analyzer_id}")
        except Exception as e:
            print(f"Failed to create analyzer '{analyzer_id}': {e}")


def analyze_content():
    """Submit each configured file to its analyzer and return the raw results.

    Returns:
        list[dict]: Each entry has 'id' (str) and 'content' (list) keys.
    """
    client = _get_cu_client()
    _ensure_cu_defaults(client)
    analyzer_content = []
    missing_input_files = []
    api_errors = []
    for analyzer in ANALYZER_CONFIGS:
        analyzer_id = analyzer["id"]
        file_location = _resolve_path(analyzer["location"])
        try:
            if not file_location.exists():
                missing_input_files.append(str(file_location))
                print(
                    f"Skipping analyzer '{analyzer_id}': input file not found at {file_location}."
                )
                continue
            with file_location.open("rb") as f:
                response = client.begin_analyze_binary(analyzer_id, f.read())
            result = response.result()
            analyzer_content.append({"id": analyzer_id, "content": result.contents})
            print(f"Analysis complete for: {analyzer_id}")
        except Exception as e:
            error_str = str(e)
            is_not_found = "ModelNotFound" in error_str or "NotFound" in error_str or "not found" in error_str.lower()
            if is_not_found:
                api_errors.append(
                    f"  - '{analyzer_id}': analyzer not registered on the service"
                )
                print(
                    f"Analyzer '{analyzer_id}' does not exist yet. Run with --create-analyzer to register it."
                )
            else:
                api_errors.append(f"  - '{analyzer_id}': {e}")
                print(f"Error analyzing '{analyzer_id}': {e}")

    if not analyzer_content:
        parts = []
        if api_errors:
            parts.append(
                "One or more analyzers failed:\n" + "\n".join(api_errors)
            )
            if any("not registered" in e for e in api_errors):
                parts.append(
                    "Run the pipeline with --create-analyzer to register missing analyzers "
                    "(requires analyzer template JSON files in the analyzer_templates/ folder)."
                )
        if missing_input_files:
            parts.append(
                "Input files not found:\n" + "\n".join(f"  - {p}" for p in missing_input_files)
                + "\nSet CU_DATA_DIR in .env or place files under contentunderstanding-rag-demo/data/."
            )
        raise RuntimeError("No analyzer results returned.\n\n" + "\n\n".join(parts))

    return analyzer_content


def process_content(analyzer_content):
    """Convert raw Content Understanding output into LangChain Document objects.

    Args:
        analyzer_content: Output from analyze_content().

    Returns:
        list[Document]: Chunked documents ready for embedding.
    """
    if not analyzer_content:
        raise ValueError(
            "No analyzer outputs were returned. Check earlier analyzer errors, endpoint/auth configuration, "
            "and run with --create-analyzer after fixing templates if needed."
        )

    output = []
    chunks_per_type = {"narrative": 0, "table": 0, "figure": 0}

    config_by_id = {cfg["id"]: cfg for cfg in ANALYZER_CONFIGS}
    prefix_by_id = {
        "cu_doc_analyzer": "Document analysis content for file: ",
        "cu_doc_layout_analyzer": "Layout-aware document analysis with structured sections, tables, and metrics for file: ",
        "cu_image_analyzer": "Chart/figure analysis content for file: ",
        "cu_audio_analyzer": "This is a json string representing an audio segment with transcription for the file located in ",
        "cu_video_analyzer": "The following is a json string representing a video segment with scene description and transcript for the file located in ",
    }

    for entry in analyzer_content:
        analyzer_id = entry.get("id")
        content_items = entry.get("content") or []
        if analyzer_id not in config_by_id or not content_items:
            continue

        location = config_by_id[analyzer_id]["location"]
        prefix = prefix_by_id.get(analyzer_id, "Analyzer output for file located in ")
        source_file = Path(location).name

        for item_index, item in enumerate(content_items):
            raw_text = _normalize_whitespace(_to_json_text(item))
            if not raw_text:
                continue

            content_type = _classify_content_type(analyzer_id, raw_text)
            splitter = _get_splitter_for_content_type(content_type)
            page_hint = _extract_page_hint(raw_text)

            base_text = prefix + str(location) + "\n\n" + raw_text
            chunks = splitter.split_text(base_text)
            if not chunks:
                continue

            for chunk_index, chunk in enumerate(chunks):
                metadata = {
                    "source_file": source_file,
                    "source_path": str(_resolve_path(location)),
                    "analyzer_id": analyzer_id,
                    "content_type": content_type,
                    "item_index": item_index,
                    "chunk_index": chunk_index,
                    "chunk_strategy": "hybrid_financial_report_v1",
                }
                if page_hint is not None:
                    metadata["page_hint"] = page_hint
                output.append(Document(page_content=chunk, metadata=metadata))
                chunks_per_type[content_type] += 1

    if not output:
        raise ValueError(
            "Analyzer outputs were received but no content chunks were produced. "
            "Verify analyzer templates and input files."
        )

    print(
        "Processed "
        f"{len(output)} chunks "
        f"(narrative={chunks_per_type['narrative']}, "
        f"table={chunks_per_type['table']}, "
        f"figure={chunks_per_type['figure']})."
    )
    return output


def embed_and_index_chunks(docs):
    """Embed document chunks and index them into Azure AI Search.

    Args:
        docs: Output from process_content().

    Returns:
        AzureSearch: Configured vector store ready for similarity queries.
    """
    aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
    )
    # Use from_documents to automatically create/update docs in the index.
    # Existing index field names id/content/content_vector/metadata are supported by
    # LangChain AzureSearch defaults when indexing Document(page_content, metadata).
    # Authentication: Use admin key if provided, otherwise use DefaultAzureCredential (Entra ID).
    # For Entra ID auth, the user's account needs "Search Index Data Contributor" role on the resource.
    vector_store = AzureSearch.from_documents(
        documents=docs,
        embedding=aoai_embeddings,
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_ADMIN_KEY,  # None = use Entra ID; string = use key auth
        index_name=AZURE_SEARCH_INDEX_NAME,
    )
    print(f"Indexed {len(docs)} chunks into '{AZURE_SEARCH_INDEX_NAME}'.")
    return vector_store


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(create_analyzer: bool = False):
    """Run the full Content Understanding pipeline.

    Args:
        create_analyzer: When True, creates the analyzers before running
            analysis. Only required on the first run or after template changes.

    Returns:
        AzureSearch: Populated vector store ready for RAG queries.
    """
    # create_analyzer = True
    if create_analyzer:
        create_analyzers()
    analyzer_content = analyze_content()
    all_splits = process_content(analyzer_content)
    vector_store = embed_and_index_chunks(all_splits)
    return vector_store


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    script_started_at = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Content Understanding RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  First run (create analyzers):  python cu_pipeline.py --create-analyzer\n"
            "  Subsequent runs:               python cu_pipeline.py\n"
            "  Pipeline only (no chat):       python cu_pipeline.py --skip-chat\n"
        ),
    )
    parser.add_argument(
        "--create-analyzer",
        action="store_true",
        default=False,
        help="Create Content Understanding analyzers before running analysis (first run only).",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        default=False,
        help="Run the pipeline only without launching the interactive chat loop.",
    )
    args = parser.parse_args()

    _vector_store = run_pipeline(create_analyzer=args.create_analyzer)

    if not args.skip_chat:
        from rag_chat import run_chat
        run_chat(_vector_store, token_provider, startup_started_at=script_started_at)