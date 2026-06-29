import json
from datetime import datetime, timezone

from langchain_core.documents import Document

from .config import CHUNK_OVERLAP, CHUNK_SIZE, INDEX_DIR, INDEX_FILES, INDEX_MANIFEST_PATH


def ensure_work_dirs() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def clear_index_files() -> None:
    ensure_work_dirs()
    for path in [*(INDEX_DIR / file_name for file_name in INDEX_FILES), INDEX_MANIFEST_PATH]:
        if path.exists():
            path.unlink()


def db_exists() -> bool:
    return INDEX_DIR.is_dir() and all((INDEX_DIR / file_name).exists() for file_name in INDEX_FILES)


def get_index_version() -> tuple[float, ...]:
    tracked_paths = [*(INDEX_DIR / file_name for file_name in INDEX_FILES), INDEX_MANIFEST_PATH]
    return tuple(path.stat().st_mtime if path.exists() else 0 for path in tracked_paths)


def _format_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def create_index_manifest(
    source_documents: list[Document],
    chunks: list[Document],
    raw_text_length: int,
) -> dict:
    sources: dict[str, dict] = {}
    for doc in source_documents:
        source = str(doc.metadata.get("source", "未知来源"))
        entry = sources.setdefault(
            source,
            {
                "source": source,
                "source_type": doc.metadata.get("source_type", "unknown"),
                "pages": set(),
                "document_units": 0,
                "text_length": 0,
            },
        )
        page = doc.metadata.get("page")
        if page:
            entry["pages"].add(page)
        entry["document_units"] += 1
        entry["text_length"] += len(doc.page_content)

    normalized_sources = []
    for entry in sources.values():
        pages = sorted(entry.pop("pages"))
        if pages:
            entry["page_count"] = len(pages)
            entry["first_page"] = pages[0]
            entry["last_page"] = pages[-1]
        normalized_sources.append(entry)

    return {
        "schema_version": 1,
        "created_at": _format_utc_now(),
        "index_dir": str(INDEX_DIR),
        "vector_store": "FAISS",
        "embedding_model": "text-embedding-v1",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "text_normalization": "NFKC",
        "chunk_count": len(chunks),
        "raw_text_length": raw_text_length,
        "source_count": len(normalized_sources),
        "sources": sorted(normalized_sources, key=lambda item: item["source"]),
        "security_note": "本地 FAISS index.pkl 只应来自本应用生成的 work/faiss_db；来源不明时请清除并重新处理文档。",
    }


def write_index_manifest(manifest: dict) -> None:
    ensure_work_dirs()
    INDEX_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_index_manifest() -> dict | None:
    if not INDEX_MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
