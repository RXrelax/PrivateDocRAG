import json
from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import rag_app.vector_store as vector_store
from rag_app.config import INDEX_BACKUP_PREFIX, INDEX_STAGING_PREFIX
from rag_app.vector_store import (
    IndexStorageError,
    cleanup_index_staging_dir,
    clear_index_files,
    create_index_manifest,
    create_index_staging_dir,
    db_exists,
    publish_staged_index,
    validate_active_index,
    write_index_manifest,
)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "vector_store": "FAISS",
        "embedding_model": "text-embedding-v1",
    }


def _configure_runtime_paths(tmp_path, monkeypatch) -> tuple[Path, Path, Path]:
    work_dir = tmp_path / "work"
    index_dir = work_dir / "faiss_db"
    error_log = work_dir / "last_embedding_error.json"
    monkeypatch.setattr("rag_app.vector_store.WORK_DIR", work_dir)
    monkeypatch.setattr("rag_app.vector_store.INDEX_DIR", index_dir)
    monkeypatch.setattr("rag_app.vector_store.EMBEDDING_ERROR_LOG_PATH", error_log)
    return work_dir, index_dir, error_log


def _write_index(index_dir: Path, marker: bytes = b"index") -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index.faiss").write_bytes(marker + b"-faiss")
    (index_dir / "index.pkl").write_bytes(marker + b"-pickle")
    (index_dir / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]


def test_create_index_manifest_has_counts_without_document_content() -> None:
    source_documents = [
        Document(
            page_content="第一页内容",
            metadata={"source": "book.pdf", "source_type": "pdf", "page": 1},
        ),
        Document(
            page_content="第二页内容",
            metadata={"source": "book.pdf", "source_type": "pdf", "page": 2},
        ),
    ]
    chunks = [
        Document(page_content="chunk 1", metadata={"source": "book.pdf", "chunk_id": 1}),
        Document(page_content="chunk 2", metadata={"source": "book.pdf", "chunk_id": 2}),
    ]

    manifest = create_index_manifest(source_documents, chunks, raw_text_length=10)

    assert manifest["schema_version"] == 1
    assert manifest["chunk_count"] == 2
    assert manifest["raw_text_length"] == 10
    assert manifest["source_count"] == 1
    assert manifest["sources"][0]["source"] == "book.pdf"
    assert manifest["sources"][0]["page_count"] == 2
    assert "第一页内容" not in str(manifest)


def test_create_index_manifest_distinguishes_same_named_uploads() -> None:
    source_documents = [
        Document(
            page_content="第一份内容",
            metadata={
                "source": "notes.txt",
                "source_type": "txt",
                "document_id": "upload-1",
            },
        ),
        Document(
            page_content="第二份内容",
            metadata={
                "source": "notes.txt",
                "source_type": "txt",
                "document_id": "upload-2",
            },
        ),
    ]

    manifest = create_index_manifest(source_documents, [], raw_text_length=10)

    assert manifest["source_count"] == 2
    assert [item["document_id"] for item in manifest["sources"]] == [
        "upload-1",
        "upload-2",
    ]


def test_db_exists_requires_valid_manifest(tmp_path, monkeypatch) -> None:
    _work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    _write_index(index_dir)

    assert db_exists()

    invalid_manifest = {**_manifest(), "schema_version": 2}
    (index_dir / "manifest.json").write_text(json.dumps(invalid_manifest), encoding="utf-8")
    assert not db_exists()


def test_db_exists_rejects_symlinked_index_file(tmp_path, monkeypatch) -> None:
    _work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    _write_index(index_dir)
    target = tmp_path / "outside.pkl"
    target.write_bytes(b"outside")
    (index_dir / "index.pkl").unlink()
    try:
        (index_dir / "index.pkl").symlink_to(target)
    except OSError:
        pytest.skip("当前平台不允许创建测试 symlink")

    assert not db_exists()


def test_clear_index_files_removes_partial_known_artifacts_only(tmp_path, monkeypatch) -> None:
    work_dir, index_dir, error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    index_dir.mkdir(parents=True)
    (index_dir / "index.faiss").write_bytes(b"partial")
    unknown = index_dir / "keep.me"
    unknown.write_text("unknown", encoding="utf-8")
    error_log.write_text("diagnostic", encoding="utf-8")

    assert clear_index_files()
    assert not (index_dir / "index.faiss").exists()
    assert not error_log.exists()
    assert unknown.read_text(encoding="utf-8") == "unknown"
    assert index_dir.is_dir()
    assert work_dir.is_dir()


def test_publish_rejects_unknown_existing_content(tmp_path, monkeypatch) -> None:
    _work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    _write_index(index_dir, b"old")
    unknown = index_dir / "keep.me"
    unknown.write_text("unknown", encoding="utf-8")
    staging_dir = create_index_staging_dir()
    _write_index(staging_dir, b"new")

    with pytest.raises(IndexStorageError, match="未知内容"):
        publish_staged_index(staging_dir)

    assert (index_dir / "index.faiss").read_bytes() == b"old-faiss"
    assert unknown.read_text(encoding="utf-8") == "unknown"
    cleanup_index_staging_dir(staging_dir)


def test_publish_switch_failure_restores_old_index(tmp_path, monkeypatch) -> None:
    work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    _write_index(index_dir, b"old")
    staging_dir = create_index_staging_dir()
    _write_index(staging_dir, b"new")
    original_rename = Path.rename

    def fail_staging_switch(path, target):
        if path == staging_dir:
            raise OSError("simulated switch failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_switch)
    with pytest.raises(IndexStorageError, match="旧索引已保留"):
        publish_staged_index(staging_dir)

    assert (index_dir / "index.faiss").read_bytes() == b"old-faiss"
    assert staging_dir.is_dir()
    assert not list(work_dir.glob(f"{INDEX_BACKUP_PREFIX}*"))
    cleanup_index_staging_dir(staging_dir)
    assert not list(work_dir.glob(f"{INDEX_STAGING_PREFIX}*"))


def test_publish_replaces_old_index_and_cleans_backup(tmp_path, monkeypatch) -> None:
    work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    _write_index(index_dir, b"old")
    staging_dir = create_index_staging_dir()
    _write_index(staging_dir, b"new")

    assert publish_staged_index(staging_dir)

    assert (index_dir / "index.faiss").read_bytes() == b"new-faiss"
    assert not list(work_dir.glob(f"{INDEX_BACKUP_PREFIX}*"))


def test_backup_cleanup_failure_keeps_published_index(tmp_path, monkeypatch) -> None:
    work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    _write_index(index_dir, b"old")
    staging_dir = create_index_staging_dir()
    _write_index(staging_dir, b"new")
    original_cleanup = vector_store._cleanup_owned_path

    def fail_backup_cleanup(path, prefix):
        if prefix == INDEX_BACKUP_PREFIX:
            raise OSError("simulated backup cleanup failure")
        return original_cleanup(path, prefix)

    monkeypatch.setattr("rag_app.vector_store._cleanup_owned_path", fail_backup_cleanup)

    assert not publish_staged_index(staging_dir)
    assert (index_dir / "index.faiss").read_bytes() == b"new-faiss"
    backups = list(work_dir.glob(f"{INDEX_BACKUP_PREFIX}*"))
    assert len(backups) == 1
    assert (backups[0] / "index.faiss").read_bytes() == b"old-faiss"


def test_real_faiss_save_publish_and_load_round_trip(tmp_path, monkeypatch) -> None:
    _work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    embeddings = DeterministicEmbeddings()
    store = FAISS.from_texts(
        ["alpha passage", "beta passage"],
        embeddings,
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    )
    staging_dir = create_index_staging_dir()
    store.save_local(str(staging_dir))
    write_index_manifest(_manifest(), staging_dir)

    assert publish_staged_index(staging_dir)
    assert validate_active_index()["vector_store"] == "FAISS"

    loaded = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    results = loaded.similarity_search("alpha", k=1)
    assert len(results) == 1
    assert results[0].page_content in {"alpha passage", "beta passage"}
