import pytest

from rag_app.config import RetrievalConfig
from langchain_core.documents import Document

from rag_app.retrieval import (
    build_keyword_weights,
    build_retriever,
    extract_container_terms,
    expand_with_neighbors,
    extract_query_terms,
    keyword_search_documents,
    looks_like_catalog,
    normalize_search_text,
    RetrievalError,
    retrieve_documents,
    unique_queries,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls = []

    def as_retriever(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


def test_build_similarity_retriever() -> None:
    db = FakeVectorStore()

    retriever = build_retriever(db, RetrievalConfig(search_type="similarity", k=5))

    assert retriever == {
        "search_type": "similarity",
        "search_kwargs": {"k": 5},
    }


def test_build_mmr_retriever_clamps_fetch_k() -> None:
    db = FakeVectorStore()

    retriever = build_retriever(
        db,
        RetrievalConfig(search_type="mmr", k=10, fetch_k=3, lambda_mult=0.25),
    )

    assert retriever == {
        "search_type": "mmr",
        "search_kwargs": {
            "k": 10,
            "fetch_k": 10,
            "lambda_mult": 0.25,
        },
    }


def test_normalize_search_text_fixes_pdf_compatibility_chars() -> None:
    assert normalize_search_text("⽩⾻精") == "白骨精"


def test_extract_query_terms_adds_chinese_number_form() -> None:
    terms = extract_query_terms("讲一下34难和76难")

    assert "三十四难" in terms
    assert "七十六难" in terms


def test_extract_container_terms_marks_outer_scope() -> None:
    assert extract_container_terms("81难中34难和76难发生了什么") == {"81难", "八十一难"}


def test_extract_query_terms_keeps_precise_title_phrases() -> None:
    terms = extract_query_terms("尸魔三戏唐三藏 圣僧恨逐美猴王")

    assert "尸魔三戏唐三藏" in terms
    assert "圣僧恨逐美猴王" in terms


def test_extract_query_terms_drops_question_filler() -> None:
    terms = extract_query_terms("三打白骨精的流程是什么？")

    assert "三打白骨精" in terms
    assert "流程是什么" not in terms
    assert "的流程" not in terms


def test_keyword_search_documents_finds_precise_terms() -> None:
    docs = [
        Document(page_content="无关内容", metadata={"source": "a", "chunk_id": 1}),
        Document(page_content="这里讲三十四难的经过", metadata={"source": "a", "chunk_id": 2}),
    ]

    hits = keyword_search_documents("34难发生了什么", docs, limit=2)

    assert hits == [docs[1]]


def test_keyword_weights_downweight_common_terms() -> None:
    documents = [
        normalize_search_text("西游记 孙悟空"),
        normalize_search_text("西游记 猪八戒"),
        normalize_search_text("西游记 沙和尚"),
        normalize_search_text("西游记 尸魔三戏唐三藏"),
    ]

    weights = build_keyword_weights(["西游记", "尸魔三戏唐三藏"], documents)

    assert weights["尸魔三戏唐三藏"] > weights["西游记"]


def test_keyword_search_documents_prefers_specific_items_over_container_scope() -> None:
    docs = [
        Document(page_content="这里反复讨论八十一难 八十一难 八十一难", metadata={"source": "a", "chunk_id": 1}),
        Document(page_content="搬运车迟三十四难", metadata={"source": "a", "chunk_id": 2}),
    ]

    hits = keyword_search_documents("八十一难中三十四难发生了什么", docs, limit=2)

    assert hits[0] == docs[1]


def test_keyword_search_documents_prefers_body_over_catalog_for_detail_query() -> None:
    docs = [
        Document(
            page_content=(
                "目录 第一回 灵根育孕源流出 第二回 悟彻菩提真妙理 "
                "第三回 四海千山皆拱伏 第四回 官封弼马心何足 "
                "第五回 乱蟠桃大圣偷丹 第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王 "
                "第二十八回 花果山群妖聚义"
            ),
            metadata={"source": "a", "chunk_id": 1},
        ),
        Document(
            page_content="第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王 却说三藏师徒次日天明收拾前进。",
            metadata={"source": "a", "chunk_id": 2},
        ),
    ]

    hits = keyword_search_documents("尸魔三戏唐三藏的详细经过", docs, limit=2)

    assert looks_like_catalog(docs[0].page_content)
    assert hits[0] == docs[1]


def test_expand_with_neighbors_adds_adjacent_chunks() -> None:
    docs = [
        Document(page_content="前文", metadata={"source": "a", "chunk_id": 1}),
        Document(page_content="命中", metadata={"source": "a", "chunk_id": 2}),
        Document(page_content="后文", metadata={"source": "a", "chunk_id": 3}),
    ]

    expanded = expand_with_neighbors([docs[1]], docs, window=1, limit=10)

    assert expanded == [docs[1], docs[0], docs[2]]


def test_expand_with_neighbors_never_crosses_sources() -> None:
    docs = [
        Document(page_content="A 末尾", metadata={"source": "a.txt", "chunk_id": 1}),
        Document(page_content="B 命中", metadata={"source": "b.txt", "chunk_id": 2}),
        Document(page_content="B 后文", metadata={"source": "b.txt", "chunk_id": 3}),
    ]

    expanded = expand_with_neighbors([docs[1]], docs, window=1, limit=3)

    assert expanded == [docs[1], docs[2]]


def test_expand_with_neighbors_distinguishes_same_source_by_document_id() -> None:
    docs = [
        Document(
            page_content="同名文件的相同片段",
            metadata={"source": "same.txt", "document_id": "upload-1", "chunk_id": 1},
        ),
        Document(
            page_content="同名文件的相同片段",
            metadata={"source": "same.txt", "document_id": "upload-2", "chunk_id": 1},
        ),
        Document(
            page_content="第二份同名文件后文",
            metadata={"source": "same.txt", "document_id": "upload-2", "chunk_id": 2},
        ),
    ]

    expanded = expand_with_neighbors([docs[1]], docs, window=1, limit=3)

    assert expanded == [docs[1], docs[2]]


def test_expand_with_neighbors_preserves_seed_when_limit_is_one() -> None:
    docs = [
        Document(page_content="前文", metadata={"source": "a", "chunk_id": 1}),
        Document(page_content="命中", metadata={"source": "a", "chunk_id": 2}),
        Document(page_content="后文", metadata={"source": "a", "chunk_id": 3}),
    ]

    assert expand_with_neighbors([docs[1]], docs, window=1, limit=1) == [docs[1]]


def test_expand_with_neighbors_dedupes_when_window_is_disabled() -> None:
    doc = Document(page_content="命中", metadata={"source": "a", "chunk_id": 1})

    assert expand_with_neighbors([doc, doc], [doc], window=0, limit=10) == [doc]


def test_retrieve_documents_dedupes_expanded_query_hits_without_neighbors(monkeypatch) -> None:
    doc = Document(page_content="命中", metadata={"source": "a", "chunk_id": 1})

    class FakeRetriever:
        def invoke(self, _query):
            return [doc]

    class FakeDatabase:
        def as_retriever(self, **_kwargs):
            return FakeRetriever()

    monkeypatch.setattr("rag_app.retrieval.load_vector_store", lambda: FakeDatabase())
    config = RetrievalConfig(
        search_type="similarity",
        k=2,
        keyword_search=False,
        neighbor_window=0,
        max_context_docs=4,
    )

    hits = retrieve_documents("原问题", config, expanded_queries=["改写问题"])

    assert hits == [doc]


def test_unique_queries_dedupes_normalized_queries() -> None:
    assert unique_queries("白骨精", [" 白骨精 ", "尸魔三戏唐三藏"]) == [
        "白骨精",
        "尸魔三戏唐三藏",
    ]


def test_retrieve_documents_wraps_unexpected_provider_errors(monkeypatch) -> None:
    def fail_to_load():
        raise ValueError("provider detail should stay internal")

    monkeypatch.setattr("rag_app.retrieval.load_vector_store", fail_to_load)

    with pytest.raises(RetrievalError, match="检索文档失败") as captured:
        retrieve_documents("问题", RetrievalConfig(search_type="similarity", k=2))

    assert isinstance(captured.value.__cause__, ValueError)
    assert "provider detail" not in str(captured.value)
