from rag_app.config import RetrievalConfig
from langchain_core.documents import Document

from rag_app.retrieval import (
    build_keyword_weights,
    build_retriever,
    chinese_under_100_to_number,
    count_section_heading_queries,
    detail_forward_window_for_query,
    expand_difficulty_queries,
    extract_container_terms,
    extract_difficulty_entries,
    extract_target_difficulty_numbers,
    expand_with_neighbors,
    extract_section_heading,
    extract_section_headings,
    extract_query_terms,
    keyword_search_each_query,
    keyword_search_documents,
    looks_like_catalog,
    looks_like_section_start,
    normalize_search_text,
    retrieve_documents,
    section_heading_queries,
    section_heading_search_documents,
    unique_queries,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls = []

    def as_retriever(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class FakeRetriever:
    def __init__(self, docs: list[Document] | None = None) -> None:
        self.docs = docs or []

    def invoke(self, _query):
        return self.docs


class FakeDocstore:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    def search(self, doc_id: str) -> Document:
        return self.docs[int(doc_id)]


class FakeRetrievalVectorStore:
    def __init__(self, docs: list[Document], vector_docs: list[Document] | None = None) -> None:
        self.index_to_docstore_id = {index: str(index) for index in range(len(docs))}
        self.docstore = FakeDocstore(docs)
        self.vector_docs = vector_docs

    def as_retriever(self, **_kwargs):
        return FakeRetriever(self.vector_docs)


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
    assert normalize_search_text("白⻣夫人和⻓老") == "白骨夫人和长老"


def test_extract_query_terms_adds_chinese_number_form() -> None:
    terms = extract_query_terms("讲一下34难和76难")

    assert "三十四难" in terms
    assert "七十六难" in terms


def test_extract_container_terms_marks_outer_scope() -> None:
    assert extract_container_terms("81难中34难和76难发生了什么") == {"81难", "八十一难"}


def test_extract_target_difficulty_numbers_excludes_outer_scope() -> None:
    assert extract_target_difficulty_numbers("八十一难中第45难和五十六难都讲了什么") == [45, 56]


def test_chinese_under_100_to_number_handles_common_forms() -> None:
    assert chinese_under_100_to_number("十") == 10
    assert chinese_under_100_to_number("十一") == 11
    assert chinese_under_100_to_number("二十") == 20
    assert chinese_under_100_to_number("五十六") == 56


def test_extract_query_terms_keeps_precise_title_phrases() -> None:
    terms = extract_query_terms("尸魔三戏唐三藏 圣僧恨逐美猴王")

    assert "尸魔三戏唐三藏" in terms
    assert "圣僧恨逐美猴王" in terms


def test_extract_query_terms_drops_question_filler() -> None:
    terms = extract_query_terms("三打白骨精的流程是什么？")

    assert "三打白骨精" in terms
    assert "流程是什么" not in terms
    assert "的流程" not in terms


def test_extract_query_terms_drops_generic_request_words() -> None:
    terms = extract_query_terms("请介绍一下三打白骨精的具体流程")

    assert "三打白骨精" in terms
    assert "介绍" not in terms
    assert "具体" not in terms
    assert "具体流程" not in terms
    assert "打白" not in terms


def test_extract_query_terms_keeps_difficulty_number_whole() -> None:
    terms = extract_query_terms("拯救疲癃五十六难")

    assert "五十六难" in terms
    assert "十六难" not in terms


def test_keyword_search_documents_finds_precise_terms() -> None:
    docs = [
        Document(page_content="无关内容", metadata={"source": "a", "chunk_id": 1}),
        Document(page_content="这里讲三十四难的经过", metadata={"source": "a", "chunk_id": 2}),
    ]

    hits = keyword_search_documents("34难发生了什么", docs, limit=2)

    assert hits == [docs[1]]


def test_keyword_search_documents_matches_pdf_radical_forms() -> None:
    docs = [
        Document(page_content="无关内容", metadata={"source": "a", "chunk_id": 1}),
        Document(page_content="这里写到白⻣夫人", metadata={"source": "a", "chunk_id": 2}),
    ]

    hits = keyword_search_documents("白骨夫人", docs, limit=2)

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
    assert docs[0] not in hits


def test_looks_like_section_start_excludes_catalog_pages() -> None:
    assert looks_like_section_start("第二十七回 尸魔三戏唐三藏 正文开始")
    assert not looks_like_section_start(
        "目录 第一回 灵根育孕源流出 第二回 悟彻菩提真妙理 "
        "第三回 四海千山皆拱伏 第四回 官封弼马心何足 "
        "第五回 乱蟠桃大圣偷丹 第六回 观音赴会问原因 "
        "第七回 八卦炉中逃大圣 第八回 我佛造经传极乐"
    )


def test_english_section_start_and_catalog_detection() -> None:
    assert looks_like_section_start("CHAPTER I. Down the Rabbit-Hole")
    assert looks_like_catalog(
        "CONTENTS CHAPTER I Down the Rabbit-Hole CHAPTER II The Pool of Tears "
        "CHAPTER III A Caucus-Race and a Long Tale"
    )
    assert not looks_like_section_start(
        "CONTENTS CHAPTER I Down the Rabbit-Hole CHAPTER II The Pool of Tears "
        "CHAPTER III A Caucus-Race and a Long Tale"
    )


def test_extract_section_heading_stops_before_body_text() -> None:
    assert (
        extract_section_heading("第二十七回  尸魔三戏唐三藏  圣僧恨逐美猴王 却说三藏师徒前进")
        == "第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王"
    )


def test_extract_section_heading_trims_chapter_body_without_fixed_delimiter() -> None:
    assert (
        extract_section_heading("第五十八回 二心搅乱大乾坤 一体难修真寂灭 这行者与沙僧拜辞了菩萨")
        == "第五十八回 二心搅乱大乾坤 一体难修真寂灭"
    )


def test_extract_english_section_heading() -> None:
    assert extract_section_heading("CHAPTER I. Down the Rabbit-Hole") == "CHAPTER I. Down the Rabbit-Hole"


def test_extract_section_headings_dedupes_titles() -> None:
    docs = [
        Document(page_content="第一回 灵根育孕源流出 诗曰正文", metadata={}),
        Document(page_content="第一回 灵根育孕源流出 诗曰正文", metadata={}),
        Document(page_content="第二回 悟彻菩提真妙理 却说正文", metadata={}),
    ]

    assert extract_section_headings(docs) == [
        "第一回 灵根育孕源流出",
        "第二回 悟彻菩提真妙理",
    ]


def test_extract_difficulty_entries_reads_list_items() -> None:
    entries = extract_difficulty_entries(
        "女国留婚四十四难 琵琶洞受苦四十五难  再贬心猿四十六难 "
        "朱紫国行医五十五难  拯救疲癃五十六难"
    )

    assert [(entry.number, entry.title) for entry in entries] == [
        (44, "女国留婚"),
        (45, "琵琶洞受苦"),
        (46, "再贬心猿"),
        (55, "朱紫国行医"),
        (56, "拯救疲癃"),
    ]


def test_expand_difficulty_queries_includes_target_and_neighbor_titles() -> None:
    docs = [
        Document(
            page_content=(
                "女国留婚四十四难 琵琶洞受苦四十五难 再贬心猿四十六难 "
                "朱紫国行医五十五难 拯救疲癃五十六难 降妖取后五十七难"
            ),
            metadata={},
        )
    ]

    queries = expand_difficulty_queries("81难中45难和56难都讲了什么", docs)

    assert "琵琶洞受苦四十五难" in queries
    assert "琵琶洞受苦" in queries
    assert "女国留婚" in queries
    assert "再贬心猿" in queries
    assert "拯救疲癃五十六难" in queries
    assert "朱紫国行医" in queries
    assert "降妖取后" in queries
    assert all("八十一难" not in query for query in queries)


def test_keyword_search_each_query_keeps_multi_target_coverage() -> None:
    docs = [
        Document(page_content="琵琶洞受苦四十五难", metadata={"chunk_id": 1}),
        Document(page_content="琵琶洞正文", metadata={"chunk_id": 2}),
        Document(page_content="朱紫国行医五十五难 拯救疲癃五十六难", metadata={"chunk_id": 3}),
        Document(page_content="朱紫国正文", metadata={"chunk_id": 4}),
    ]

    hits = keyword_search_each_query(
        ["琵琶洞受苦", "朱紫国行医"],
        docs,
        per_query_limit=2,
        total_limit=4,
    )

    assert [doc.metadata["chunk_id"] for doc in hits] == [1, 2, 3, 4]


def test_expand_with_neighbors_adds_adjacent_chunks() -> None:
    docs = [
        Document(page_content="前文", metadata={"source": "a", "chunk_id": 1}),
        Document(page_content="命中", metadata={"source": "a", "chunk_id": 2}),
        Document(page_content="后文", metadata={"source": "a", "chunk_id": 3}),
    ]

    expanded = expand_with_neighbors([docs[1]], docs, window=1, limit=10)

    assert expanded == docs


def test_expand_with_neighbors_extends_section_start_forward() -> None:
    docs = [
        Document(page_content="前文", metadata={"source": "a", "chunk_id": 1}),
        Document(page_content="第二十七回 尸魔三戏唐三藏 正文开始", metadata={"source": "a", "chunk_id": 2}),
        Document(page_content="后文一", metadata={"source": "a", "chunk_id": 3}),
        Document(page_content="后文二", metadata={"source": "a", "chunk_id": 4}),
        Document(page_content="后文三", metadata={"source": "a", "chunk_id": 5}),
        Document(page_content="后文四", metadata={"source": "a", "chunk_id": 6}),
    ]

    expanded = expand_with_neighbors([docs[1]], docs, window=1, limit=5, detail_forward_window=3)

    assert expanded == docs[1:5]


def test_expand_with_neighbors_does_not_expand_catalog_pages() -> None:
    docs = [
        Document(page_content="前文", metadata={"source": "a", "chunk_id": 1}),
        Document(
            page_content=(
                "目录 第一回 灵根育孕源流出 第二回 悟彻菩提真妙理 "
                "第三回 四海千山皆拱伏 第四回 官封弼马心何足 "
                "第五回 乱蟠桃大圣偷丹 第六回 观音赴会问原因 "
                "第七回 八卦炉中逃大圣 第八回 我佛造经传极乐"
            ),
            metadata={"source": "a", "chunk_id": 2},
        ),
        Document(page_content="后文", metadata={"source": "a", "chunk_id": 3}),
    ]

    expanded = expand_with_neighbors([docs[1]], docs, window=1, limit=10)

    assert expanded == [docs[1]]


def test_unique_queries_dedupes_normalized_queries() -> None:
    assert unique_queries("白骨精", [" 白骨精 ", "尸魔三戏唐三藏"]) == [
        "白骨精",
        "尸魔三戏唐三藏",
    ]


def test_detail_forward_window_balances_multiple_section_queries() -> None:
    assert (
        detail_forward_window_for_query(
            "真假美猴王的起因和结果是什么",
            [
                "第五十七回 真行者落伽山诉苦 假猴王水帘洞誊文",
                "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
            ],
        )
        == 8
    )
    assert (
        detail_forward_window_for_query(
            "请讲讲这两个章节的经过",
            [
                "第五十七回 真行者落伽山诉苦 假猴王水帘洞誊文",
                "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
            ],
        )
        == 3
    )
    assert detail_forward_window_for_query(
        "请介绍一下三打白骨精的具体流程",
        ["第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王"],
    ) == 8


def test_count_section_heading_queries_ignores_plain_rewrites() -> None:
    assert count_section_heading_queries(["真假美猴王", "CHAPTER I. Down the Rabbit-Hole"]) == 1


def test_section_heading_queries_filters_formal_headings() -> None:
    assert section_heading_queries(
        [
            "真假美猴王",
            "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
            "CHAPTER I. Down the Rabbit-Hole",
        ]
    ) == [
        "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
        "CHAPTER I. Down the Rabbit-Hole",
    ]


def test_section_heading_search_documents_skips_catalog_matches() -> None:
    docs = [
        Document(
            page_content=(
                "目录 第一回 灵根育孕源流出 第二回 悟彻菩提真妙理 "
                "第三回 四海千山皆拱伏 第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王"
            ),
            metadata={"chunk_id": "catalog"},
        ),
        Document(
            page_content="第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王 却说三藏师徒前进",
            metadata={"chunk_id": "body"},
        ),
    ]

    hits = section_heading_search_documents(
        ["第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王"],
        docs,
        total_limit=3,
    )

    assert [doc.metadata["chunk_id"] for doc in hits] == ["body"]


def test_retrieve_documents_prioritizes_section_heading_anchors(monkeypatch) -> None:
    docs = [
        Document(page_content="第五十七回 真行者落伽山诉苦 假猴王水帘洞誊文", metadata={"chunk_id": 57}),
        Document(page_content="第五十八回 二心搅乱大乾坤 一体难修真寂灭", metadata={"chunk_id": 58}),
    ]
    noise = Document(page_content="泛泛背景内容", metadata={"chunk_id": 999})
    monkeypatch.setattr(
        "rag_app.retrieval.load_vector_store",
        lambda: FakeRetrievalVectorStore(docs, vector_docs=[noise]),
    )

    hits = retrieve_documents(
        "真假美猴王",
        RetrievalConfig(
            search_type="similarity",
            k=2,
            keyword_search=True,
            neighbor_window=0,
            max_context_docs=2,
        ),
        [
            "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
            "第五十七回 真行者落伽山诉苦 假猴王水帘洞誊文",
        ],
    )

    assert [doc.metadata["chunk_id"] for doc in hits] == [58, 57]
