import pytest
from langchain_core.documents import Document

from rag_app.indexing import IndexBuildError, _embed_slice_with_retry


class FlakyEmbeddings:
    def __init__(self) -> None:
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if len(texts) > 1:
            raise RuntimeError("batch too large")
        return [[float(len(texts[0]))]]


class BrokenEmbeddings:
    def embed_documents(self, _texts):
        raise RuntimeError("bad text")


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

    with pytest.raises(IndexBuildError, match="向量化片段 1 失败"):
        _embed_slice_with_retry(BrokenEmbeddings(), ["cannot embed"], chunks, 0, 1)

    assert log_path.exists()
    assert "cannot embed" not in log_path.read_text(encoding="utf-8")
