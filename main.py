from fastmcp import FastMCP
import os
import uuid
from typing import List, Dict, Any
import time
import threading
import cgi
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

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

# Simple upload gateway (Approach 3: pre-signed one-tap link)
UPLOAD_ENABLED = os.getenv("UPLOAD_ENABLED", "true").lower() in {"1", "true", "yes"}
UPLOAD_HOST = os.getenv("UPLOAD_HOST", "0.0.0.0")
UPLOAD_PORT = int(os.getenv("UPLOAD_PORT", 9001))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
UPLOAD_PUBLIC_BASE = os.getenv("UPLOAD_PUBLIC_BASE", f"http://localhost:{UPLOAD_PORT}")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# token -> {"collection": str, "expires_at": float}
UPLOAD_TOKENS: Dict[str, Dict[str, Any]] = {}

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


# -----------------------------
# Minimal stdlib upload server
# -----------------------------
class UploadHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "u":
            token = parts[1]
            record = UPLOAD_TOKENS.get(token)
            if not record or time.time() > record.get("expires_at", 0):
                self._send_response(410, "<h1>Link expired</h1>")
                return
            html = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>Upload PDF</title></head>
<body>
  <h3>Upload PDF to collection: {record['collection']}</h3>
  <form method='POST' enctype='multipart/form-data'>
    <input type='file' name='file' accept='application/pdf' required />
    <br/><br/>
    <label>Chunk size: <input type='number' name='chunk_size' value='1000'/></label>
    <label>Overlap: <input type='number' name='chunk_overlap' value='200'/></label>
    <br/><br/>
    <button type='submit'>Upload</button>
  </form>
</body></html>
"""
            self._send_response(200, html, content_type="text/html; charset=utf-8")
            return
        self._send_response(404, "Not found")

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "u":
            token = parts[1]
            record = UPLOAD_TOKENS.get(token)
            if not record or time.time() > record.get("expires_at", 0):
                self._send_response(410, "<h1>Link expired</h1>")
                return

            ctype, pdict = cgi.parse_header(self.headers.get('content-type', ''))
            if ctype != 'multipart/form-data':
                self._send_response(400, "Expected multipart/form-data")
                return
            pdict['boundary'] = pdict['boundary'].encode('utf-8') if isinstance(pdict.get('boundary'), str) else pdict.get('boundary')
            pdict['CONTENT-LENGTH'] = int(self.headers.get('content-length', '0'))
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'}, keep_blank_values=True)

            file_field = form['file'] if 'file' in form else None
            if not file_field or not getattr(file_field, 'filename', None):
                self._send_response(400, "No file uploaded")
                return

            chunk_size = int(form.getfirst('chunk_size', '1000'))
            chunk_overlap = int(form.getfirst('chunk_overlap', '200'))
            safe_name = os.path.basename(file_field.filename)
            dest_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}-{safe_name}")

            # Save file
            with open(dest_path, 'wb') as out:
                data = file_field.file.read()
                out.write(data)

            # Ingest
            try:
                summary = rag_ingest_pdf(
                    collection=record['collection'],
                    file_path=dest_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except Exception as e:
                self._send_response(400, f"<h1>Ingest failed</h1><pre>{e}</pre>", content_type="text/html; charset=utf-8")
                return

            # Invalidate token after single use
            UPLOAD_TOKENS.pop(token, None)

            html = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>Uploaded</title></head>
<body>
  <h3>Upload successful</h3>
  <p>Saved as: {summary['file']}</p>
  <p>Collection: {summary['collection']}</p>
  <p>Chunks added: {summary['chunks_added']}</p>
</body></html>
"""
            self._send_response(200, html, content_type="text/html; charset=utf-8")
            return
        self._send_response(404, "Not found")

    def log_message(self, format, *args):  # suppress console spam
        return

    def _send_response(self, status_code: int, body: str, content_type: str = "text/plain; charset=utf-8"):
        body_bytes = body.encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)


def start_upload_server():
    if not UPLOAD_ENABLED:
        return
    server = HTTPServer((UPLOAD_HOST, UPLOAD_PORT), UploadHTTPRequestHandler)
    server.serve_forever()


@mcp.tool()
def rag_create_upload_link(collection: str, expires_in_seconds: int = 600) -> Dict[str, Any]:
    """Create a one-time upload link for a PDF that auto-ingests into a collection.

    Returns a URL the user can tap to upload directly from the phone.
    """
    if expires_in_seconds < 60:
        expires_in_seconds = 60
    token = uuid.uuid4().hex
    UPLOAD_TOKENS[token] = {
        "collection": collection,
        "expires_at": time.time() + expires_in_seconds,
    }
    url = f"{UPLOAD_PUBLIC_BASE}/u/{token}"
    return {"upload_url": url, "expires_at": int(UPLOAD_TOKENS[token]["expires_at"]) }

if __name__ == "__main__":
    # Start upload server in background
    threading.Thread(target=start_upload_server, daemon=True).start()
    # Start MCP server
    mcp.run(transport="streamable-http")
