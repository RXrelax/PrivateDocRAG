from langchain_core.documents import Document

from rag_app.sources import (
    build_retrieval_entries,
    build_source_entries,
    extract_cited_source_numbers,
    format_source_label,
    preview_text,
)


def test_format_source_label_includes_page_and_chunk() -> None:
    doc = Document(
        page_content="content",
        metadata={"source": "book.pdf", "page": 3, "chunk_id": 9},
    )

    assert format_source_label(doc) == "book.pdf / 第 3 页 / 片段 9"


def test_extract_cited_source_numbers_dedupes_and_filters_range() -> None:
    answer = "第一句 [来源2]，第二句 [source: 3]，重复 [来源2]，越界 [来源9]。"

    assert extract_cited_source_numbers(answer, max_sources=3) == [2, 3]


def test_preview_text_compacts_whitespace_and_truncates() -> None:
    assert preview_text("a\n\n b\tc", limit=20) == "a b c"
    assert preview_text("abcdef", limit=4) == "abcd..."


def test_build_source_entries_only_lists_explicit_citations() -> None:
    docs = [
        Document(page_content="first source", metadata={"source": "a.txt"}),
        Document(page_content="second source", metadata={"source": "b.txt"}),
    ]

    entries = build_source_entries("没有显式引用", docs)

    assert entries == []


def test_build_retrieval_entries_lists_all_sources() -> None:
    docs = [
        Document(page_content="first source", metadata={"source": "a.txt"}),
        Document(page_content="second source", metadata={"source": "b.txt"}),
    ]

    entries = build_retrieval_entries(docs)

    assert [entry["number"] for entry in entries] == [1, 2]
    assert [entry["label"] for entry in entries] == ["a.txt", "b.txt"]
    assert [entry["chars"] for entry in entries] == [12, 13]
