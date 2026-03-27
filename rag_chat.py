import os
import time

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_API_VERSION = os.getenv("AZURE_OPENAI_CHAT_API_VERSION") or "2024-08-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

_PROMPT_STR = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""


def setup_rag_chain(vector_store, token_provider):
    """Build and return a RAG chain from the given vector store.

    Args:
        vector_store: Populated AzureSearch vector store from cu_pipeline.
        token_provider: Bearer token provider for Azure OpenAI authentication.

    Returns:
        A runnable LangChain RAG chain.
    """
    retriever = vector_store.as_retriever(search_type="similarity", k=3)
    prompt = ChatPromptTemplate.from_template(_PROMPT_STR)
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_CHAT_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        azure_ad_token_provider=token_provider,
        temperature=0.7,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def run_chat(vector_store, token_provider, startup_started_at=None):
    """Start the interactive RAG chat loop.

    Args:
        vector_store: Populated AzureSearch vector store from cu_pipeline.
        token_provider: Bearer token provider for Azure OpenAI authentication.
        startup_started_at: Optional perf-counter timestamp from script start.
    """
    rag_chain = setup_rag_chain(vector_store, token_provider)
    print("RAG chat ready. Press Enter with no input to quit.")
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
        answer = rag_chain.invoke(query)
        print(answer)
        query_elapsed = time.perf_counter() - query_started_at
        print(f"[timer] Query to answer returned: {query_elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Entry point (standalone use)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from cu_pipeline import run_pipeline, token_provider
    script_started_at = time.perf_counter()
    _vector_store = run_pipeline(create_analyzer=False)
    run_chat(_vector_store, token_provider, startup_started_at=script_started_at)