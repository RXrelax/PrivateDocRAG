import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_app.config import INDEX_STAGING_PREFIX
from rag_app.indexing import (
    IndexBuildError,
    _embed_slice_with_retry,
    _save_vector_store_atomically,
    process_documents,
)


class FlakyEmbeddings:
    def __init__(self) -> None:
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if len(texts) > 1:
            raise RuntimeError("batch too large")
        return [[float(len(texts[0]))]]


class BrokenEmbeddings:
    def __init__(self, message="bad text") -> None:
        self.message = message

    def embed_documents(self, _texts):
        raise RuntimeError(self.message)


class SavingStore:
    def save_local(self, index_dir) -> None:
        target = Path(index_dir)
        (target / "index.faiss").write_bytes(b"new-faiss")
        (target / "index.pkl").write_bytes(b"new-pickle")


class FailingSaveStore:
    def save_local(self, index_dir) -> None:
        target = Path(index_dir)
        (target / "index.faiss").write_bytes(b"partial")
        raise OSError("simulated save failure")


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
    monkeypatch.setattr("rag_app.indexing.INDEX_DIR", index_dir)
    return work_dir, index_dir, error_log


def _write_old_index(index_dir: Path) -> None:
    index_dir.mkdir(parents=True)
    (index_dir / "index.faiss").write_bytes(b"old-faiss")
    (index_dir / "index.pkl").write_bytes(b"old-pickle")
    (index_dir / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")


def test_embed_slice_splits_failed_batches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("rag_app.indexing.EMBEDDING_MAX_RETRIES", 0)
    monkeypatch.setattr("rag_app.indexing.EMBEDDING_ERROR_LOG_PATH", tmp_path / "error.json")
    embeddings = FlakyEmbeddings()
    chunks = [
        Document(page_content="aa", metadata={"source": "a.txt", "chunk_id": 1}),
        Document(page_content="bbb", metadata={"source": "a.txt", "chunk_id": 2}),
    ]
    texts = [chunk.page_content for chunk in chunks]

    vectors = _embed_slice_with_retry(embeddings, texts, chunks, 0, 2)

    assert vectors == [[2.0], [3.0]]
    assert ["aa", "bbb"] in embeddings.calls
    assert ["aa"] in embeddings.calls
    assert ["bbb"] in embeddings.calls


def test_embed_slice_reports_single_chunk_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("rag_app.indexing.EMBEDDING_MAX_RETRIES", 0)
    log_path = tmp_path / "last_embedding_error.json"
    monkeypatch.setattr("rag_app.indexing.EMBEDDING_ERROR_LOG_PATH", log_path)
    chunks = [
        Document(
            page_content="cannot embed",
            metadata={"source": "a.txt", "source_type": "txt", "chunk_id": 1},
        )
    ]

    sensitive_marker = "provider-sensitive-marker"
    with pytest.raises(IndexBuildError, match="向量化片段 1 失败"):
        _embed_slice_with_retry(
            BrokenEmbeddings(sensitive_marker),
            ["cannot embed"],
            chunks,
            0,
            1,
        )

    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "cannot embed" not in log_text
    assert sensitive_marker not in log_text
    assert "原始异常未记录" in log_text


def test_save_failure_preserves_old_index(tmp_path, monkeypatch) -> None:
    work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    _write_old_index(index_dir)

    with pytest.raises(IndexBuildError, match="保存 FAISS 索引失败"):
        _save_vector_store_atomically(FailingSaveStore(), _manifest())

    assert (index_dir / "index.faiss").read_bytes() == b"old-faiss"
    assert (index_dir / "index.pkl").read_bytes() == b"old-pickle"
    assert not list(work_dir.glob(f"{INDEX_STAGING_PREFIX}*"))


def test_staging_cleanup_failure_does_not_mask_save_error(tmp_path, monkeypatch) -> None:
    _work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    _write_old_index(index_dir)

    def fail_cleanup(_path):
        raise OSError("cleanup failed")

    monkeypatch.setattr(
        "rag_app.indexing.cleanup_index_staging_dir",
        fail_cleanup,
    )

    with pytest.raises(IndexBuildError, match="保存 FAISS 索引失败"):
        _save_vector_store_atomically(FailingSaveStore(), _manifest())

    assert (index_dir / "index.faiss").read_bytes() == b"old-faiss"


def test_manifest_failure_preserves_old_index(tmp_path, monkeypatch) -> None:
    work_dir, index_dir, _error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    _write_old_index(index_dir)

    def fail_manifest_write(_manifest_payload, _index_dir):
        raise OSError("simulated manifest failure")

    monkeypatch.setattr("rag_app.indexing.write_index_manifest", fail_manifest_write)
    with pytest.raises(IndexBuildError, match="manifest"):
        _save_vector_store_atomically(SavingStore(), _manifest())

    assert (index_dir / "index.faiss").read_bytes() == b"old-faiss"
    assert (index_dir / "index.pkl").read_bytes() == b"old-pickle"
    assert not list(work_dir.glob(f"{INDEX_STAGING_PREFIX}*"))


def test_successful_process_clears_stale_embedding_error_log(tmp_path, monkeypatch) -> None:
    _work_dir, index_dir, error_log = _configure_runtime_paths(tmp_path, monkeypatch)
    error_log.parent.mkdir(parents=True)
    error_log.write_text("stale", encoding="utf-8")
    source_documents = [
        Document(page_content="正文", metadata={"source": "a.txt", "source_type": "txt"})
    ]
    chunks = [
        Document(
            page_content="正文",
            metadata={"source": "a.txt", "source_type": "txt", "chunk_id": 1},
        )
    ]
    monkeypatch.setattr(
        "rag_app.indexing.read_uploaded_documents",
        lambda *_args, **_kwargs: source_documents,
    )
    monkeypatch.setattr("rag_app.indexing.split_documents", lambda _documents: chunks)
    monkeypatch.setattr(
        "rag_app.indexing.build_vector_store",
        lambda *_args, **_kwargs: SavingStore(),
    )
    monkeypatch.setattr("rag_app.indexing.clear_vector_store_cache", lambda: None)

    result = process_documents([])

    assert result.chunk_count == 1
    assert not error_log.exists()
    assert (index_dir / "manifest.json").is_file()


def test_post_publish_cleanup_failures_become_warnings(monkeypatch) -> None:
    source_documents = [
        Document(page_content="正文", metadata={"source": "a.txt", "source_type": "txt"})
    ]
    chunks = [
        Document(
            page_content="正文",
            metadata={"source": "a.txt", "source_type": "txt", "chunk_id": 1},
        )
    ]
    warnings = []
    cache_clear_calls = []

    def fail_log_cleanup():
        raise OSError("cleanup failed")

    monkeypatch.setattr("rag_app.indexing.ensure_work_dirs", lambda: None)
    monkeypatch.setattr(
        "rag_app.indexing.read_uploaded_documents",
        lambda *_args, **_kwargs: source_documents,
    )
    monkeypatch.setattr("rag_app.indexing.split_documents", lambda _documents: chunks)
    monkeypatch.setattr(
        "rag_app.indexing.build_vector_store",
        lambda *_args, **_kwargs: SavingStore(),
    )
    monkeypatch.setattr(
        "rag_app.indexing._save_vector_store_atomically",
        lambda *_args: False,
    )
    monkeypatch.setattr("rag_app.indexing.clear_embedding_error_log", fail_log_cleanup)
    monkeypatch.setattr(
        "rag_app.indexing.clear_vector_store_cache",
        lambda: cache_clear_calls.append(True),
    )

    result = process_documents([], warn=warnings.append)

    assert result.chunk_count == 1
    assert cache_clear_calls == [True]
    assert any("旧备份或临时目录" in message for message in warnings)
    assert any("诊断日志" in message for message in warnings)
