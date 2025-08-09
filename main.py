from fastmcp import FastMCP
import os
import uuid
from typing import List, Dict, Any

from dotenv import load_dotenv

# RAG
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()


PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
mcp = FastMCP("rag-mcp", host=HOST, port=PORT)

PHONE_NUMBER = os.getenv("PHONE_NUMBER")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")


@mcp.tool()
def validate(token: str) -> str:
    """Returns the Phone Number of the MCP server Owner.
    Args:
        token: Bearer token to validate
    Returns:
        A string of the Phone Number of the MCP server Owner.
    """
    if token == BEARER_TOKEN and PHONE_NUMBER:
        return f"91{PHONE_NUMBER}"
    raise ValueError("Invalid token")


CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.getcwd(), "vector_store"))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Persistent Chroma client for collection management
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

def get_or_create_collection(collection: str):
    return chroma_client.get_or_create_collection(name=collection)

def get_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=api_key)

def get_vectorstore(collection: str) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

def load_and_split_pdf(file_path: str, chunk_size: int, chunk_overlap: int):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(200, chunk_size),
        chunk_overlap=max(0, min(chunk_overlap, chunk_size // 2)),
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

@mcp.tool()
def rag_list_collections() -> List[str]:
    """List available RAG collections (vector stores)."""
    cols = chroma_client.list_collections()
    return [c.name for c in cols]

@mcp.tool()
def rag_create_collection(collection: str) -> str:
    """Create a new RAG collection if it doesn't exist.

    Args:
        collection: Name of the collection
    Returns:
        Confirmation message
    """
    get_or_create_collection(collection)
    return f"Collection '{collection}' is ready."

@mcp.tool()
def rag_delete_collection(collection: str) -> str:
    """Delete a RAG collection and all its vectors."""
    chroma_client.delete_collection(name=collection)
    return f"Collection '{collection}' deleted."

@mcp.tool()
def rag_ingest_pdf(
    collection: str,
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """Ingest a local PDF file into a vector DB collection.

    Steps: read PDF, chunk, embed, and upsert.

    Args:
        collection: Target collection name
        file_path: Absolute or relative path to a local PDF file
        chunk_size: Number of words per chunk
        chunk_overlap: Number of overlapping words between chunks

    Returns:
        Summary with counts and collection name
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Ensure collection exists at DB level
    get_or_create_collection(collection)

    # Load and split with LangChain
    splits = load_and_split_pdf(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not splits:
        raise ValueError("No text extracted from PDF.")

    # Augment metadata
    for i, d in enumerate(splits):
        d.metadata = {**d.metadata, "source": os.path.basename(file_path), "chunk_index": i}

    # Upsert into Chroma via LangChain
    vs = get_vectorstore(collection)
    vs.add_documents(splits)
    vs.persist()

    return {
        "collection": collection,
        "file": os.path.basename(file_path),
        "chunks_added": len(splits),
        "persist_path": CHROMA_DIR,
    }

@mcp.tool()
def rag_query(
    collection: str,
    question: str,
    k: int = 5,
) -> Dict[str, Any]:
    """Retrieve top-k relevant chunks for a question from a collection.

    Args:
        collection: Name of the collection to search
        question: The user question to retrieve context for
        k: Number of results to return

    Returns:
        A dict containing retrieved documents, metadata, distances, and a combined context string
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    # Ensure collection exists
    get_or_create_collection(collection)

    vs = get_vectorstore(collection)
    try:
        doc_score_pairs = vs.similarity_search_with_relevance_scores(question, k=max(1, k))
    except Exception:
        # Fallback if relevance scores not supported by installed version
        docs_only = vs.similarity_search(question, k=max(1, k))
        doc_score_pairs = [(d, None) for d in docs_only]

    documents = [d.page_content for d, _ in doc_score_pairs]
    metadatas = [d.metadata for d, _ in doc_score_pairs]
    scores = [s for _, s in doc_score_pairs]
    combined_context = "\n\n".join(documents) if documents else ""

    return {
        "collection": collection,
        "question": question,
        "k": k,
        "documents": documents,
        "metadatas": metadatas,
        "relevance_scores": scores,
        "context": combined_context,
    }

@mcp.tool()
def rag_answer(
    collection: str,
    question: str,
    k: int = 5,
    model: str | None = None,
) -> Dict[str, Any]:
    """Answer a question using retrieval-augmented generation.

    Requires environment variable `OPENAI_API_KEY`. Optionally set `OPENAI_MODEL`.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    # Build retriever
    get_or_create_collection(collection)
    vs = get_vectorstore(collection)
    retriever = vs.as_retriever(search_kwargs={"k": max(1, k)})

    # Build LLM
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=api_key)

    # Build RetrievalQA chain
    from langchain.chains import RetrievalQA

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    result = qa.invoke({"query": question})
    answer_text = result.get("result", "")
    source_docs = result.get("source_documents", []) or []
    sources = [getattr(d, "metadata", {}) for d in source_docs]

    return {
        "answer": answer_text,
        "sources": sources,
        "used_model": model_name,
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
