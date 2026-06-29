from langchain_core.documents import Document

from rag_app.vector_store import create_index_manifest


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

