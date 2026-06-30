from langchain_core.documents import Document

from rag_app.qa import (
    build_answer_style_instruction,
    build_context,
    build_history_text,
    build_query_hint_text,
    build_section_heading_prompt,
    build_user_prompt,
    expand_section_neighbor_queries,
    expand_section_heading_queries,
    expand_search_queries,
    find_matching_section_headings,
    merge_expanded_queries,
    normalize_message_content,
    order_section_queries_for_result_intent,
    parse_expanded_queries,
    parse_section_heading_suggestions,
    query_matches_section_heading,
)


def test_build_context_formats_numbered_sources() -> None:
    docs = [
        Document(
            page_content="相关内容",
            metadata={"source": "a.txt", "chunk_id": 1},
        )
    ]

    assert build_context(docs).startswith("[来源 1：a.txt / 片段 1]\n相关内容")


def test_build_history_text_uses_recent_messages() -> None:
    history = [{"role": "user", "content": str(index)} for index in range(10)]

    text = build_history_text(history, max_messages=2)

    assert "用户：8" in text
    assert "用户：9" in text
    assert "用户：7" not in text


def test_normalize_message_content_handles_text_parts() -> None:
    content = [{"text": "hello"}, {"other": "world"}]

    assert normalize_message_content(content) == "hello\n{'other': 'world'}"


def test_build_answer_style_instruction_defaults_to_detailed() -> None:
    assert "充分展开" in build_answer_style_instruction("详细解释")
    assert build_answer_style_instruction("unknown") == build_answer_style_instruction("详细解释")


def test_build_user_prompt_includes_citation_and_synthesis_rules() -> None:
    prompt = build_user_prompt(
        question="问题？",
        context="[来源 1：a]\n内容",
        history_text="（无）",
        answer_style="详细解释",
        expanded_queries=["相关标题"],
    )

    assert "每个关键结论尽量标注来源" in prompt
    assert "综合多个来源" in prompt
    assert "先直接回答问题" in prompt
    assert "相关标题" not in prompt
    assert "检索线索：" not in prompt
    assert "不要用常识补完长答案" in prompt
    assert "没有出现在检索上下文中，不要把它写进答案" in prompt
    assert "不要用“必然”“应当就是”等措辞" in prompt
    assert "辨别地点" in prompt
    assert "用户问题：问题？" in prompt


def test_build_query_hint_text_marks_empty_queries() -> None:
    assert build_query_hint_text([]) == "（无）"
    assert build_query_hint_text(["尸魔三戏唐三藏"]) == "- 尸魔三戏唐三藏"


def test_parse_expanded_queries_strips_numbering_and_limits() -> None:
    content = "1. 尸魔三戏唐三藏\n2. 第二十七回 白骨夫人\n3. 三打白骨精\n4. extra"

    assert parse_expanded_queries(content, "三打白骨精") == [
        "尸魔三戏唐三藏",
        "第二十七回 白骨夫人",
        "extra",
    ]


def test_query_matches_section_heading_accepts_partial_overlap() -> None:
    headings = ["第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王"]

    assert query_matches_section_heading("尸魔三戏唐三藏", headings)
    assert not query_matches_section_heading("三打白骨精", headings)


def test_find_matching_section_headings_returns_known_headings() -> None:
    headings = [
        "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
        "第六回 观音赴会问原因 小圣施威降大圣",
    ]

    assert find_matching_section_headings(
        ["二心搅乱大乾坤", "第六回 观音赴会问原因"],
        headings,
    ) == headings


def test_parse_section_heading_suggestions_only_returns_known_headings() -> None:
    headings = [
        "第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王",
        "第二十八回 花果山群妖聚义 黑松林三藏逢魔",
    ]
    content = (
        "1. 第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王。\n"
        "2. 三打白骨精\n"
        "3. 第二十八回 花果山群妖聚义 黑松林三藏逢魔"
    )

    assert parse_section_heading_suggestions(content, headings) == headings


def test_build_section_heading_prompt_constrains_model_to_candidates() -> None:
    prompt = build_section_heading_prompt(
        "三打白骨精",
        ["白骨精故事"],
        ["第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王"],
    )

    assert "只能原样输出候选标题" in prompt
    assert "已有检索改写" in prompt
    assert "第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王" in prompt


def test_expand_section_heading_queries_selects_from_local_headings(monkeypatch) -> None:
    heading = "第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王"

    class FakeLLM:
        def __init__(self) -> None:
            self.prompts = []

        def invoke(self, messages):
            self.prompts.append(messages[0].content)
            return type("Response", (), {"content": heading})()

    fake_llm = FakeLLM()
    monkeypatch.setattr("rag_app.qa.get_section_headings", lambda: [heading])
    monkeypatch.setattr("rag_app.qa.get_llm", lambda: fake_llm)

    assert expand_section_heading_queries("三打白骨精", ["白骨精流程"]) == [heading]
    assert "候选标题" in fake_llm.prompts[0]


def test_expand_section_heading_queries_uses_single_existing_heading_match(monkeypatch) -> None:
    heading = "第二十七回 尸魔三戏唐三藏 圣僧恨逐美猴王"

    class FailingLLM:
        def invoke(self, _messages):
            raise AssertionError("LLM should not be called")

    monkeypatch.setattr("rag_app.qa.get_section_headings", lambda: [heading])
    monkeypatch.setattr("rag_app.qa.get_llm", lambda: FailingLLM())

    assert expand_section_heading_queries("三打白骨精", ["尸魔三戏唐三藏"]) == [heading]


def test_expand_section_heading_queries_disambiguates_conflicting_heading_matches(monkeypatch) -> None:
    target = "第五十八回 二心搅乱大乾坤 一体难修真寂灭"
    wrong = "第六回 观音赴会问原因 小圣施威降大圣"

    class FakeLLM:
        def invoke(self, messages):
            assert target in messages[0].content
            assert wrong in messages[0].content
            return type("Response", (), {"content": target})()

    monkeypatch.setattr("rag_app.qa.get_section_headings", lambda: [target, wrong])
    monkeypatch.setattr("rag_app.qa.get_llm", lambda: FakeLLM())

    assert expand_section_heading_queries(
        "真假美猴王的起因和结果是什么",
        ["二心搅乱大乾坤", "第六回 观音赴会问原因"],
    ) == [target]


def test_expand_section_neighbor_queries_adds_adjacent_headings_for_span_questions(monkeypatch) -> None:
    headings = [
        "第五十七回 真行者落伽山诉苦 假猴王水帘洞誊文",
        "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
        "第五十九回 唐三藏路阻火焰山 孙行者一调芭蕉扇",
    ]

    monkeypatch.setattr("rag_app.qa.get_section_headings", lambda: headings)

    assert expand_section_neighbor_queries(
        "真假美猴王的起因和结果是什么",
        ["第五十八回 二心搅乱大乾坤 一体难修真寂灭"],
    ) == [
        "第五十七回 真行者落伽山诉苦 假猴王水帘洞誊文",
        "第五十九回 唐三藏路阻火焰山 孙行者一调芭蕉扇",
    ]


def test_expand_section_neighbor_queries_skips_simple_title_questions(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_app.qa.get_section_headings",
        lambda: [
            "第五十七回 真行者落伽山诉苦 假猴王水帘洞誊文",
            "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
        ],
    )

    assert expand_section_neighbor_queries(
        "第58回叫什么",
        ["第五十八回 二心搅乱大乾坤 一体难修真寂灭"],
    ) == []


def test_order_section_queries_for_result_intent_prefers_later_headings(monkeypatch) -> None:
    headings = [
        "第五十七回 真行者落伽山诉苦 假猴王水帘洞誊文",
        "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
    ]
    monkeypatch.setattr("rag_app.qa.get_section_headings", lambda: headings)

    assert order_section_queries_for_result_intent("真假美猴王的起因和结果是什么", headings) == [
        headings[1],
        headings[0],
    ]
    assert order_section_queries_for_result_intent("真假美猴王的起因是什么", headings) == headings


def test_merge_expanded_queries_dedupes_against_original_question() -> None:
    assert merge_expanded_queries(
        "三打白骨精",
        [" 三打白骨精 ", "白骨夫人"],
        ["白骨夫人", "尸魔三戏唐三藏"],
    ) == ["白骨夫人", "尸魔三戏唐三藏"]


def test_expand_search_queries_skips_section_mapping_when_difficulty_queries_exist(monkeypatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return type(
                "Response",
                (),
                {"content": "第四十五回 三清观大圣留名\n女儿国蝎子精"},
            )()

    def fail_section_queries(*_args, **_kwargs):
        raise AssertionError("Section heading mapping should not run for numbered-list matches")

    monkeypatch.setattr("rag_app.qa.get_llm", lambda: FakeLLM())
    monkeypatch.setattr("rag_app.qa.expand_difficulty_queries", lambda _question: ["琵琶洞受苦"])
    monkeypatch.setattr("rag_app.qa.expand_section_heading_queries", fail_section_queries)

    assert expand_search_queries("45难讲了什么") == [
        "琵琶洞受苦",
        "女儿国蝎子精",
    ]


def test_expand_search_queries_skips_neighbor_expansion_for_multiple_section_matches(monkeypatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return type("Response", (), {"content": "真假美猴王"})()

    section_queries = [
        "第五十七回 真行者落伽山诉苦 假猴王水帘洞誊文",
        "第五十八回 二心搅乱大乾坤 一体难修真寂灭",
    ]

    def fail_neighbor_expansion(_question, _selected_headings, window=1):
        raise AssertionError("Neighbor expansion should not run when multiple sections are selected")

    monkeypatch.setattr("rag_app.qa.get_llm", lambda: FakeLLM())
    monkeypatch.setattr("rag_app.qa.expand_difficulty_queries", lambda _question: [])
    monkeypatch.setattr("rag_app.qa.expand_section_heading_queries", lambda *_args, **_kwargs: section_queries)
    monkeypatch.setattr("rag_app.qa.expand_section_neighbor_queries", fail_neighbor_expansion)

    assert expand_search_queries("真假美猴王的起因和结果是什么") == [
        *section_queries,
        "真假美猴王",
    ]


def test_expand_search_queries_can_be_disabled() -> None:
    assert expand_search_queries("问题", enabled=False) == []
