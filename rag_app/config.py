from dataclasses import dataclass
from pathlib import Path


WORK_DIR = Path("work")
INDEX_DIR = WORK_DIR / "faiss_db"
INDEX_FILES = ("index.faiss", "index.pkl")
INDEX_MANIFEST_FILE = "manifest.json"
INDEX_MANIFEST_PATH = INDEX_DIR / INDEX_MANIFEST_FILE
EMBEDDING_ERROR_LOG_PATH = WORK_DIR / "last_embedding_error.json"

DEFAULT_RETRIEVAL_K = 8
DEFAULT_FETCH_K = 40
DEFAULT_LAMBDA_MULT = 0.5
DEFAULT_KEYWORD_K = 6
DEFAULT_NEIGHBOR_WINDOW = 1
DEFAULT_MAX_CONTEXT_DOCS = 14
DEFAULT_QUERY_EXPANSION = True
EMBEDDING_BATCH_SIZE = 25
EMBEDDING_MAX_RETRIES = 2
EMBEDDING_RETRY_BASE_SECONDS = 1.5

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class RetrievalConfig:
    search_type: str
    k: int
    fetch_k: int = DEFAULT_FETCH_K
    lambda_mult: float = DEFAULT_LAMBDA_MULT
    keyword_search: bool = True
    keyword_k: int = DEFAULT_KEYWORD_K
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW
    max_context_docs: int = DEFAULT_MAX_CONTEXT_DOCS
    query_expansion: bool = DEFAULT_QUERY_EXPANSION


@dataclass(frozen=True)
class ProcessDocumentsResult:
    chunk_count: int
    raw_text_length: int
    manifest: dict
