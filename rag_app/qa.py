from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .config import RetrievalConfig
from .model_services import get_llm
from .retrieval import (
    expand_difficulty_queries,
    get_section_headings,
    has_result_intent,
    looks_like_section_start,
    normalize_search_text,
    retrieve_documents,
)
from .sources import format_source_label


SECTION_NEIGHBOR_INTENT_TERMS = {
    "起因",
    "结果",
    "前因",
    "后果",
    "来龙去脉",
    "完整",
    "从头到尾",
    "前后",
    "经过",
}


SYSTEM_PROMPT = """你是一个本地文档问答助手。你的首要任务是基于检索到的文档片段回答用户问题。

规则：
1. 优先使用检索上下文，不要把没有依据的内容说成文档事实。
2. 如果上下文能支持答案，要给出清楚、完整、有层次的解释。
3. 如果多个来源分别提供了不同信息，要综合它们，而不是只依赖第一个来源。
4. 对关键事实、情节、人物关系、章节判断，应在相关句子后标注 [来源N]。
5. 如果上下文不足，要明确说明缺口；除非用户明确要求，不要用常识补完长答案。
6. 不要补写检索上下文没有出现的专名、药方、战斗细节、人物关系、辨别地点或结局，即使你知道常识答案。
7. 检索线索只用于检索，不是文档事实；如果某个词只出现在检索线索中、没有出现在上下文中，不要把它写进答案或建议。
8. 后续检索建议只能使用用户问题或检索上下文中已经出现的词。
9. 不要机械复述检索片段，要把信息整理成用户容易理解的回答。"""


def normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def history_entry_to_text(entry) -> str:
    if isinstance(entry, dict):
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
    else:
        role = getattr(entry, "role", None) or getattr(entry, "type", entry.__class__.__name__)
        content = getattr(entry, "content", str(entry))

    role_labels = {
        "user": "用户",
        "human": "用户",
        "assistant": "助手",
        "ai": "助手",
    }
    label = role_labels.get(str(role).lower(), str(role))
    return f"{label}：{normalize_message_content(content)}"


def build_history_text(history: list, max_messages: int = 8) -> str:
    if not history:
        return "（无）"
    return "\n".join(history_entry_to_text(entry) for entry in history[-max_messages:])


def build_context(source_documents: list[Document]) -> str:
    if not source_documents:
        return "（没有检索到相关上下文。）"

    sections = []
    for index, doc in enumerate(source_documents, start=1):
        content = doc.page_content.strip()
        if not content:
            continue
        sections.append(f"[来源 {index}：{format_source_label(doc)}]\n{content}")
    return "\n\n".join(sections) if sections else "（没有检索到可用文本。）"


def build_answer_style_instruction(answer_style: str) -> str:
    instructions = {
        "详细解释": (
            "回答详略：请给出充分展开的回答。除非用户明确要求简短，否则不要只给摘要。"
            "应包含：直接结论、关键依据、必要背景、过程拆解、可能的不确定点。"
            "如果问题适合分点，请使用分点结构。"
        ),
        "标准回答": (
            "回答详略：给出中等长度回答。覆盖关键事实、必要解释和主要依据，避免过短，也不要无关扩写。"
        ),
        "简短回答": (
            "回答详略：保持简洁，只回答核心结论和最必要依据。"
        ),
    }
    return instructions.get(answer_style, instructions["详细解释"])


def build_query_hint_text(expanded_queries: list[str] | None) -> str:
    if not expanded_queries:
        return "（无）"
    return "\n".join(f"- {query}" for query in expanded_queries)


def build_user_prompt(
    question: str,
    context: str,
    history_text: str,
    answer_style: str,
    expanded_queries: list[str] | None = None,
) -> str:
    return (
        "请根据下面的检索上下文回答用户问题。\n\n"
        "回答要求：\n"
        "- 先直接回答问题，再展开解释。\n"
        "- 如果问题涉及故事情节、章节、人物关系或事件顺序，请尽量分点说明。\n"
        "- 每个关键结论尽量标注来源，例如 [来源1] 或 [来源2][来源5]。\n"
        "- 不要为了引用而引用；只有确实支撑该句的来源才标注。\n"
        "- 如果检索结果彼此分散，请综合多个来源，不要只围绕第一个来源作答。\n"
        "- 如果检索上下文没有覆盖问题，要说明缺口和可继续检索的方向，不要用常识补完长答案。\n"
        "- 不要补充检索上下文没有出现的专名、药方、战斗细节、人物关系、辨别地点或结局，即使你知道常识答案。\n"
        "- 不要用“必然”“应当就是”等措辞推断上下文没有直接覆盖的结局或事实。\n"
        "- 如果上下文只间接暗示结果，不要补写具体未出现的辨别地点、真实身份或收伏方式。\n"
        "- 检索线索只解释系统为何尝试这些查询，不是文档事实，不能当作来源引用。\n"
        "- 如果某个词只出现在检索线索中、没有出现在检索上下文中，不要把它写进答案或后续检索建议。\n"
        "- 后续检索建议只能使用用户问题或检索上下文中已经出现的词。\n\n"
        f"{build_answer_style_instruction(answer_style)}\n\n"
        f"对话历史：\n{history_text}\n\n"
        f"检索上下文：\n{context}\n\n"
        f"用户问题：{question}"
    )


def parse_expanded_queries(content: str, original_question: str, limit: int = 3) -> list[str]:
    original_normalized = normalize_search_text(original_question)
    queries = []
    seen = {original_normalized}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        line = line.lstrip("-*0123456789.、) ）\t ")
        line = line.strip("`'\"“”")
        normalized_line = normalize_search_text(line)
        if not normalized_line or normalized_line in seen or len(line) > 80:
            continue
        seen.add(normalized_line)
        queries.append(line)
        if len(queries) >= limit:
            break
    return queries


def query_matches_section_heading(query: str, headings: list[str]) -> bool:
    normalized_query = normalize_search_text(query)
    if not normalized_query:
        return False
    for heading in headings:
        normalized_heading = normalize_search_text(heading)
        if normalized_heading and (
            normalized_query in normalized_heading or normalized_heading in normalized_query
        ):
            return True
    return False


def find_matching_section_headings(
    queries: list[str],
    headings: list[str],
    limit: int = 3,
) -> list[str]:
    matches = []
    seen = set()
    for query in queries:
        normalized_query = normalize_search_text(query)
        if not normalized_query:
            continue
        for heading in headings:
            normalized_heading = normalize_search_text(heading)
            if not normalized_heading:
                continue
            if normalized_query not in normalized_heading and normalized_heading not in normalized_query:
                continue
            if normalized_heading in seen:
                continue
            seen.add(normalized_heading)
            matches.append(heading)
            if len(matches) >= limit:
                return matches
    return matches


def parse_section_heading_suggestions(
    content: str,
    headings: list[str],
    limit: int = 3,
) -> list[str]:
    normalized_to_heading = {normalize_search_text(heading): heading for heading in headings}
    suggestions = []
    seen = set()

    for raw_line in content.splitlines():
        line = raw_line.strip()
        line = line.lstrip("-*0123456789.、) ）\t ")
        line = line.strip("`'\"“”。；;，, ")
        normalized_line = normalize_search_text(line)
        if not normalized_line:
            continue

        matched_heading = normalized_to_heading.get(normalized_line)
        if not matched_heading:
            for normalized_heading, heading in normalized_to_heading.items():
                if normalized_heading in normalized_line or normalized_line in normalized_heading:
                    matched_heading = heading
                    break

        if not matched_heading:
            continue

        normalized_heading = normalize_search_text(matched_heading)
        if normalized_heading in seen:
            continue
        seen.add(normalized_heading)
        suggestions.append(matched_heading)
        if len(suggestions) >= limit:
            break

    return suggestions


def build_section_heading_prompt(
    question: str,
    existing_queries: list[str],
    headings: list[str],
) -> str:
    return (
        "下面是本地文档中抽取到的章节或小节标题。"
        "请从候选标题中选择最可能对应用户问题的 1 到 3 个标题。\n"
        "要求：\n"
        "- 只能原样输出候选标题，不要改写，不要解释。\n"
        "- 如果没有合适标题，输出空。\n"
        "- 用户问题可能使用通俗名称、概括说法或民间叫法；候选标题是文档中的正式表达。\n\n"
        f"用户问题：{question}\n\n"
        f"已有检索改写：\n{build_query_hint_text(existing_queries)}\n\n"
        "候选标题：\n"
        + "\n".join(f"- {heading}" for heading in headings)
    )


def expand_section_heading_queries(
    question: str,
    existing_queries: list[str],
    enabled: bool = True,
) -> list[str]:
    if not enabled:
        return []

    try:
        headings = get_section_headings()
    except Exception:
        return []

    if not headings:
        return []

    matching_headings = find_matching_section_headings([question, *existing_queries], headings)
    if len(matching_headings) == 1:
        return matching_headings

    prompt = build_section_heading_prompt(question, existing_queries, headings)
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
    except Exception:
        return matching_headings[:1] if has_section_neighbor_intent(question) else matching_headings
    suggestions = parse_section_heading_suggestions(normalize_message_content(response.content), headings)
    if suggestions:
        return suggestions
    return matching_headings[:1] if has_section_neighbor_intent(question) else matching_headings


def has_section_neighbor_intent(question: str) -> bool:
    normalized = normalize_search_text(question)
    return any(term in normalized for term in SECTION_NEIGHBOR_INTENT_TERMS)


def expand_section_neighbor_queries(
    question: str,
    selected_headings: list[str],
    window: int = 1,
) -> list[str]:
    if not selected_headings or window <= 0 or not has_section_neighbor_intent(question):
        return []

    try:
        headings = get_section_headings()
    except Exception:
        return []

    normalized_headings = [normalize_search_text(heading) for heading in headings]
    selected = {normalize_search_text(heading) for heading in selected_headings}
    seen = set(selected)
    queries = []

    for selected_heading in selected:
        if not selected_heading:
            continue
        for index, normalized_heading in enumerate(normalized_headings):
            if not normalized_heading:
                continue
            if selected_heading not in normalized_heading and normalized_heading not in selected_heading:
                continue
            start = max(0, index - window)
            end = min(len(headings), index + window + 1)
            for neighbor_index in range(start, end):
                neighbor = normalized_headings[neighbor_index]
                if not neighbor or neighbor in seen:
                    continue
                seen.add(neighbor)
                queries.append(headings[neighbor_index])
            break

    return queries


def order_section_queries_for_result_intent(question: str, section_queries: list[str]) -> list[str]:
    if len(section_queries) <= 1 or not has_result_intent(question):
        return section_queries

    try:
        headings = get_section_headings()
    except Exception:
        return section_queries

    normalized_headings = [normalize_search_text(heading) for heading in headings]

    def heading_position(query: str) -> int:
        normalized_query = normalize_search_text(query)
        for index, normalized_heading in enumerate(normalized_headings):
            if normalized_query and (
                normalized_query in normalized_heading or normalized_heading in normalized_query
            ):
                return index
        return -1

    return sorted(
        section_queries,
        key=lambda query: heading_position(query),
        reverse=True,
    )


def merge_expanded_queries(
    original_question: str,
    *query_lists: list[str],
    limit: int = 10,
) -> list[str]:
    original_normalized = normalize_search_text(original_question)
    merged = []
    seen = {original_normalized}
    for queries in query_lists:
        for query in queries:
            normalized_query = normalize_search_text(query)
            if not normalized_query or normalized_query in seen:
                continue
            seen.add(normalized_query)
            merged.append(query)
            if len(merged) >= limit:
                return merged
    return merged


def expand_search_queries(question: str, enabled: bool = True) -> list[str]:
    if not enabled:
        return []

    try:
        difficulty_queries = expand_difficulty_queries(question)
    except Exception:
        difficulty_queries = []

    prompt = (
        "请为下面这个文档检索问题生成 1 到 3 个可用于检索的中文改写查询。\n"
        "要求：\n"
        "- 只输出查询词，每行一个。\n"
        "- 优先输出可能出现在原文标题、目录、索引、章节名或小节名中的正式叫法。\n"
        "- 如果原问题使用通俗名称、概括说法或民间叫法，优先改写成文档目录可能使用的标题式表达。\n"
        "- 可以包含同义说法、常见别名、人物/事件专名、编号的中文/阿拉伯写法。\n"
        "- 不要只重复书名、文档名、“内容”“过程”“情节”等宽泛词。\n"
        "- 不要回答问题，不要解释。\n\n"
        f"原问题：{question}"
    )
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        rewritten_queries = parse_expanded_queries(normalize_message_content(response.content), question)
    except Exception:
        rewritten_queries = []

    if difficulty_queries:
        rewritten_queries = [
            query for query in rewritten_queries if not looks_like_section_start(query)
        ]
        section_queries = []
        section_neighbor_queries = []
    else:
        section_input_queries = merge_expanded_queries(question, rewritten_queries)
        section_queries = expand_section_heading_queries(question, section_input_queries, enabled=enabled)
        section_queries = order_section_queries_for_result_intent(question, section_queries)
        section_neighbor_queries = (
            [] if len(section_queries) > 1 else expand_section_neighbor_queries(question, section_queries)
        )
    return merge_expanded_queries(
        question,
        difficulty_queries,
        section_queries,
        section_neighbor_queries,
        rewritten_queries,
    )


def answer_question(
    question: str,
    retrieval_config: RetrievalConfig,
    history: list | None = None,
    answer_style: str = "详细解释",
    progress: Callable[[int, str, str], None] | None = None,
) -> tuple[str, list[Document], list[str]]:
    def notify(percent: int, status: str, detail: str = "") -> None:
        if progress:
            progress(percent, status, detail)

    notify(10, "分析问题", "准备生成检索线索")
    expanded_queries = expand_search_queries(question, retrieval_config.query_expansion)
    if expanded_queries:
        notify(30, "生成检索线索", f"已生成 {len(expanded_queries)} 条检索线索")
    else:
        notify(30, "生成检索线索", "没有生成额外检索线索，将直接使用原问题检索")

    notify(45, "检索上下文", "正在合并向量检索、关键词补召回和相邻片段")
    source_documents = retrieve_documents(question, retrieval_config, expanded_queries)
    notify(65, "检索上下文", f"已选出 {len(source_documents)} 个候选片段")

    notify(75, "组织上下文", "正在整理来源编号和对话历史")
    context = build_context(source_documents)
    history_text = build_history_text(history or [])
    user_prompt = build_user_prompt(
        question,
        context,
        history_text,
        answer_style,
        expanded_queries=expanded_queries,
    )
    notify(88, "生成回答", "正在调用模型整理答案")
    try:
        response = get_llm().invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        raise RuntimeError("调用 DeepSeek 生成回答失败，请检查网络、额度或 DEEPSEEK_API_KEY。") from exc
    notify(100, "回答完成", "已完成检索和回答生成")
    return normalize_message_content(response.content), source_documents, expanded_queries
