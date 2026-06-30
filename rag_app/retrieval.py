from dataclasses import dataclass
import math
import re

from langchain_core.documents import Document

from .config import RetrievalConfig
from .model_services import load_vector_store
from .text_normalization import normalize_cjk_text, normalize_searchable_text


LOW_SIGNAL_TERMS = {
    "什么",
    "一下",
    "讲一",
    "讲下",
    "介绍",
    "具体",
    "说明",
    "相关",
    "主要",
    "流程",
    "过程",
    "发生",
    "故事",
    "情节",
    "梳理",
    "详细",
    "内容",
    "怎么",
    "如何",
    "为何",
    "为什么",
    "哪个",
    "哪些",
    "以及",
    "其中",
    "这个",
    "那个",
    "上下文",
}

DETAIL_INTENT_TERMS = {
    "流程",
    "过程",
    "发生",
    "故事",
    "情节",
    "经过",
    "详细",
    "梳理",
    "展开",
    "讲了什么",
    "讲什么",
    "讲的是",
    "讲一下",
    "讲讲",
    "怎么",
    "如何",
    "原因",
    "起因",
    "结果",
    "结局",
    "前因",
    "后果",
    "来龙去脉",
}

RESULT_INTENT_TERMS = {
    "结果",
    "结局",
    "后果",
    "最终",
    "后来",
    "收场",
}

LIST_INTENT_TERMS = {
    "目录",
    "列表",
    "清单",
    "有哪些",
    "多少",
    "名称",
    "标题",
}

CHINESE_DIGITS = "零一二三四五六七八九"
CHINESE_DIGIT_VALUES = {digit: index for index, digit in enumerate(CHINESE_DIGITS)}
CHINESE_SECTION_PATTERN = r"第?[一二三四五六七八九十百千万0-9]{1,5}[回章节难]"
WESTERN_SECTION_PATTERN = r"(?:chapter|section|part|book)\s+(?:[ivxlcdm]+|\d+|[a-z]+)\.?"
SINGLE_SECTION_DETAIL_FORWARD_WINDOW = 8
MULTI_SECTION_DETAIL_FORWARD_WINDOW = 3
TERM_SPLIT_TOKENS = sorted(
    {
        *LOW_SIGNAL_TERMS,
        "请",
        "帮我",
        "给我",
        "告诉我",
        "的",
        "了",
        "是",
        "吗",
        "呢",
        "吧",
    },
    key=len,
    reverse=True,
)


@dataclass(frozen=True)
class DifficultyEntry:
    number: int
    title: str


def normalize_search_text(text: str) -> str:
    return normalize_searchable_text(text)


def number_to_chinese_under_100(number: int) -> str | None:
    if not 0 < number < 100:
        return None
    if number < 10:
        return CHINESE_DIGITS[number]
    tens, ones = divmod(number, 10)
    if tens == 1:
        prefix = "十"
    else:
        prefix = f"{CHINESE_DIGITS[tens]}十"
    return prefix if ones == 0 else f"{prefix}{CHINESE_DIGITS[ones]}"


def chinese_under_100_to_number(text: str) -> int | None:
    if not text or text == "零":
        return None
    if text == "十":
        return 10
    if "十" not in text:
        value = CHINESE_DIGIT_VALUES.get(text)
        return value if value and value < 100 else None

    tens_text, ones_text = text.split("十", 1)
    if tens_text:
        tens = CHINESE_DIGIT_VALUES.get(tens_text)
        if not tens:
            return None
    else:
        tens = 1

    if ones_text:
        ones = CHINESE_DIGIT_VALUES.get(ones_text)
        if ones is None:
            return None
    else:
        ones = 0
    return tens * 10 + ones


def difficulty_number_from_text(text: str) -> int | None:
    if text.isdigit():
        number = int(text)
        return number if 0 < number < 100 else None
    return chinese_under_100_to_number(text)


def split_meaningful_runs(run: str) -> list[str]:
    splitter = "|".join(re.escape(token) for token in TERM_SPLIT_TOKENS)
    return [part for part in re.split(splitter, run) if len(part) >= 2]


def extract_container_difficulty_numbers(question: str) -> set[int]:
    normalized = normalize_search_text(question)
    numbers: set[int] = set()
    for match in re.finditer(r"(\d{1,3})\s*难\s*(?:中|里|当中|里面)", normalized):
        number = int(match.group(1))
        if 0 < number < 100:
            numbers.add(number)
    for match in re.finditer(
        r"([一二三四五六七八九十]{1,3})难\s*(?:中|里|当中|里面)",
        normalized,
    ):
        number = chinese_under_100_to_number(match.group(1))
        if number:
            numbers.add(number)
    return numbers


def extract_container_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for number in extract_container_difficulty_numbers(question):
        terms.add(f"{number}难")
        chinese_number = number_to_chinese_under_100(number)
        if chinese_number:
            terms.add(f"{chinese_number}难")
    return terms


def extract_target_difficulty_numbers(question: str) -> list[int]:
    normalized = normalize_search_text(question)
    numbers: list[int] = []
    seen = set()

    for match in re.finditer(r"第?(\d{1,3})\s*难", normalized):
        number = int(match.group(1))
        if 0 < number < 100 and number not in seen:
            seen.add(number)
            numbers.append(number)

    for match in re.finditer(r"第?([一二三四五六七八九十]{1,3})难", normalized):
        number = chinese_under_100_to_number(match.group(1))
        if number and number not in seen:
            seen.add(number)
            numbers.append(number)

    container_numbers = extract_container_difficulty_numbers(question)
    return [number for number in numbers if number not in container_numbers]


def extract_query_terms(question: str) -> list[str]:
    normalized = normalize_search_text(question)
    terms: set[str] = set()

    for match in re.finditer(r"(\d{1,2})\s*难", normalized):
        number = int(match.group(1))
        terms.add(f"{number}难")
        chinese_number = number_to_chinese_under_100(number)
        if chinese_number:
            terms.add(f"{chinese_number}难")

    for match in re.finditer(r"[一二三四五六七八九十]{1,3}难", normalized):
        terms.add(match.group(0))

    for run in re.findall(r"[a-z0-9\u4e00-\u9fff]+", normalized):
        if re.fullmatch(r"[a-z0-9]+", run):
            if len(run) >= 2:
                terms.add(run)
            continue

        numeric_difficulty_count = len(re.findall(r"\d{1,2}\s*难", run)) + len(
            re.findall(r"[一二三四五六七八九十]{1,3}难", run)
        )
        if numeric_difficulty_count >= 2:
            continue

        for part in split_meaningful_runs(run):
            if len(part) <= 24:
                terms.add(part)
            if re.search(r"(\d{1,3}|[一二三四五六七八九十百千万]{1,5})难", part):
                continue
            min_width = 2 if len(part) <= 3 else 3
            for width in range(min_width, min(6, len(part)) + 1):
                for start in range(0, len(part) - width + 1):
                    terms.add(part[start : start + width])

    return sorted(
        (term for term in terms if len(term) >= 2 and not is_low_signal_term(term)),
        key=lambda term: (-len(term), term),
    )


def build_retriever(db, config: RetrievalConfig):
    if config.search_type == "mmr":
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.k,
                "fetch_k": max(config.fetch_k, config.k),
                "lambda_mult": config.lambda_mult,
            },
        )
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.k},
    )


def get_ordered_documents(db) -> list[Document]:
    documents = []
    for index in sorted(db.index_to_docstore_id):
        doc = db.docstore.search(db.index_to_docstore_id[index])
        if isinstance(doc, Document):
            documents.append(doc)
    return documents


def document_key(doc: Document) -> tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("chunk_id"),
        doc.metadata.get("page"),
        doc.page_content[:80],
    )


def is_low_signal_term(term: str) -> bool:
    if term in LOW_SIGNAL_TERMS:
        return True
    if re.fullmatch(r"第?[0-9一二三四五六七八九十百千万]+", term):
        return True
    if len(term) <= 4 and any(signal in term for signal in LOW_SIGNAL_TERMS):
        return True
    return False


def has_detail_intent(question: str) -> bool:
    normalized = normalize_search_text(question)
    return any(term in normalized for term in DETAIL_INTENT_TERMS)


def has_list_intent(question: str) -> bool:
    normalized = normalize_search_text(question)
    return any(term in normalized for term in LIST_INTENT_TERMS)


def has_result_intent(question: str) -> bool:
    normalized = normalize_search_text(question)
    return any(term in normalized for term in RESULT_INTENT_TERMS)


def looks_like_catalog(text: str) -> bool:
    normalized = normalize_search_text(text)
    heading_count = len(
        re.findall(CHINESE_SECTION_PATTERN, normalized)
    ) + len(
        re.findall(WESTERN_SECTION_PATTERN, normalized, flags=re.IGNORECASE)
    )
    return heading_count >= 8 or (
        ("目录" in normalized[:120] or "contents" in normalized[:160])
        and heading_count >= 3
    )


def looks_like_section_start(text: str) -> bool:
    normalized = normalize_search_text(text)
    if looks_like_catalog(normalized):
        return False
    return bool(
        re.match(rf"^{CHINESE_SECTION_PATTERN}\s*", normalized)
        or re.match(rf"^{WESTERN_SECTION_PATTERN}\s*", normalized, flags=re.IGNORECASE)
    )


def extract_section_heading(text: str) -> str | None:
    if not looks_like_section_start(text):
        return None

    compact_text = re.sub(r"\s+", " ", normalize_cjk_text(text)).strip()
    match = re.match(rf"^({CHINESE_SECTION_PATTERN})\s*(.+)", compact_text)
    if match:
        marker = match.group(1)
        title = match.group(2)
        for delimiter in (" 却说", " 诗曰", " 话说", " 且说", " 原来", " 却道", " 话表", " ["):
            position = title.find(delimiter)
            if position > 0:
                title = title[:position]
                break
        if marker.endswith("回"):
            title_parts = title.split()
            if len(title_parts) >= 2:
                title = " ".join(title_parts[:2])
        return f"{marker} {title}"[:80].strip(" ,，。；;")

    match = re.match(rf"^({WESTERN_SECTION_PATTERN})\s*(.*)", compact_text, flags=re.IGNORECASE)
    if not match:
        return None
    heading = f"{match.group(1)} {match.group(2)}"
    for delimiter in (" 却说", " 诗曰", " 话说", " 且说", " 原来", " 却道"):
        position = heading.find(delimiter)
        if position > 0:
            heading = heading[:position]
            break
    return heading[:80].strip(" ,，。；;")


def extract_section_headings(documents: list[Document], limit: int = 120) -> list[str]:
    headings = []
    seen = set()
    for doc in documents:
        heading = extract_section_heading(doc.page_content)
        if not heading:
            continue
        normalized = normalize_search_text(heading)
        if normalized in seen:
            continue
        seen.add(normalized)
        headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def get_section_headings(limit: int = 120) -> list[str]:
    return extract_section_headings(get_ordered_documents(load_vector_store()), limit=limit)


def extract_difficulty_entries(text: str) -> list[DifficultyEntry]:
    compact_text = re.sub(r"\s+", " ", normalize_cjk_text(text)).strip()
    entries: list[DifficultyEntry] = []
    for match in re.finditer(r"([^\s，。；;、:：]{2,14}?)(\d{1,3}|[一二三四五六七八九十]{1,3})难", compact_text):
        title = match.group(1).strip()
        number = difficulty_number_from_text(match.group(2))
        if not number or not title:
            continue
        entries.append(DifficultyEntry(number=number, title=title))
    return entries


def build_difficulty_queries(entries: list[DifficultyEntry], target_numbers: list[int]) -> list[str]:
    target_set = set(target_numbers)
    queries = []
    seen = set()

    def add_query(query: str) -> None:
        normalized = normalize_search_text(query)
        if normalized and normalized not in seen:
            seen.add(normalized)
            queries.append(query)

    for index, entry in enumerate(entries):
        if entry.number not in target_set:
            continue

        chinese_number = number_to_chinese_under_100(entry.number)
        if chinese_number:
            add_query(f"{entry.title}{chinese_number}难")
        add_query(entry.title)

        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(entries):
                add_query(entries[neighbor].title)

    return queries


def expand_difficulty_queries(question: str, documents: list[Document] | None = None) -> list[str]:
    target_numbers = extract_target_difficulty_numbers(question)
    if not target_numbers:
        return []

    if documents is None:
        documents = get_ordered_documents(load_vector_store())

    queries = []
    seen = set()
    for doc in documents:
        entries = extract_difficulty_entries(doc.page_content)
        if not entries:
            continue
        for query in build_difficulty_queries(entries, target_numbers):
            normalized = normalize_search_text(query)
            if normalized in seen:
                continue
            seen.add(normalized)
            queries.append(query)
    return queries


def build_keyword_weights(
    terms: list[str],
    normalized_documents: list[str],
    downweighted_terms: set[str] | None = None,
) -> dict[str, float]:
    if not normalized_documents:
        return {}

    document_count = len(normalized_documents)
    downweighted_terms = downweighted_terms or set()
    weights = {}
    for term in terms:
        document_frequency = sum(1 for text in normalized_documents if term in text)
        if not document_frequency:
            continue

        idf = math.log((document_count + 1) / (document_frequency + 0.5)) + 1
        length_weight = len(term) ** 2
        precision_boost = 1.0
        if len(term) >= 6:
            precision_boost = 1.6
        elif len(term) == 2:
            precision_boost = 0.6
        if re.search(r"(\d{1,3}|[一二三四五六七八九十百千万]{1,5})难", term):
            precision_boost *= 2.0

        container_penalty = 0.25 if term in downweighted_terms else 1.0
        weights[term] = length_weight * idf * precision_boost * container_penalty
    return weights


def keyword_search_documents(
    question: str,
    documents: list[Document],
    limit: int,
) -> list[Document]:
    terms = extract_query_terms(question)
    if not terms or limit <= 0:
        return []

    normalized_documents = [normalize_search_text(doc.page_content) for doc in documents]
    downweighted_terms = extract_container_terms(question)
    keyword_weights = build_keyword_weights(
        terms,
        normalized_documents,
        downweighted_terms=downweighted_terms,
    )
    if not keyword_weights:
        return []

    document_frequencies = {
        term: sum(1 for text in normalized_documents if term in text)
        for term in keyword_weights
    }
    rare_anchor_limit = max(12, int(len(normalized_documents) * 0.02))
    anchor_terms = {
        term
        for term, frequency in document_frequencies.items()
        if len(term) >= 3 and term not in downweighted_terms and frequency <= rare_anchor_limit
    }

    wants_detail = has_detail_intent(question)
    wants_list = has_list_intent(question)

    scored = []
    for position, (doc, text) in enumerate(zip(documents, normalized_documents)):
        score = 0.0
        matched_terms = 0
        matched_anchor = False
        for term, weight in keyword_weights.items():
            count = text.count(term)
            if count:
                score += count * weight
                matched_terms += 1
                if term in anchor_terms:
                    matched_anchor = True
        if score:
            score *= 1 + min(matched_terms / 5, 1.0)
            if matched_anchor:
                score *= 1.8
            if wants_detail and not wants_list and looks_like_catalog(text):
                score *= 0.45
            scored.append((score, position, doc, matched_anchor))

    scored.sort(key=lambda item: (-item[0], item[1]))
    if anchor_terms:
        anchored_scored = [item for item in scored if item[3]]
        if anchored_scored:
            scored = anchored_scored
    if wants_detail and not wants_list:
        body_scored = [item for item in scored if not looks_like_catalog(item[2].page_content)]
        if body_scored:
            scored = body_scored
    return [doc for _score, _position, doc, _matched_anchor in scored[:limit]]


def keyword_search_each_query(
    queries: list[str],
    documents: list[Document],
    per_query_limit: int,
    total_limit: int,
) -> list[Document]:
    if not queries or per_query_limit <= 0 or total_limit <= 0:
        return []

    results = []
    for query in queries:
        results.extend(keyword_search_documents(query, documents, per_query_limit))
        if len(dedupe_documents(results)) >= total_limit:
            break
    return dedupe_documents(results)[:total_limit]


def section_heading_search_documents(
    queries: list[str],
    documents: list[Document],
    total_limit: int,
) -> list[Document]:
    if not queries or total_limit <= 0:
        return []

    results = []
    seen = set()
    for query in queries:
        normalized_query = normalize_search_text(query)
        if not normalized_query:
            continue
        for doc in documents:
            heading = extract_section_heading(doc.page_content)
            if not heading:
                continue
            normalized_heading = normalize_search_text(heading)
            if normalized_query not in normalized_heading and normalized_heading not in normalized_query:
                continue
            key = document_key(doc)
            if key not in seen:
                seen.add(key)
                results.append(doc)
            break
        if len(results) >= total_limit:
            break
    return results


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen = set()
    unique = []
    for doc in documents:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def expand_with_neighbors(
    documents: list[Document],
    ordered_documents: list[Document],
    window: int,
    limit: int,
    detail_forward_window: int = 0,
) -> list[Document]:
    if not documents:
        return []
    if window <= 0 and detail_forward_window <= 0:
        return documents[:limit]

    positions = {document_key(doc): index for index, doc in enumerate(ordered_documents)}
    expanded = []
    for doc in documents:
        position = positions.get(document_key(doc))
        if position is None:
            expanded.append(doc)
            continue
        if looks_like_catalog(doc.page_content):
            expanded.append(doc)
            continue
        is_section_start = looks_like_section_start(doc.page_content)
        forward_window = window
        if detail_forward_window > window and is_section_start:
            forward_window = detail_forward_window
        start = position if is_section_start else max(0, position - window)
        end = min(len(ordered_documents), position + forward_window + 1)
        expanded.extend(ordered_documents[start:end])
    return dedupe_documents(expanded)[:limit]


def unique_queries(question: str, expanded_queries: list[str] | None = None) -> list[str]:
    queries = []
    seen = set()
    for query in [question, *(expanded_queries or [])]:
        normalized = normalize_search_text(query)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        queries.append(query)
    return queries


def count_section_heading_queries(queries: list[str] | None) -> int:
    return sum(1 for query in queries or [] if looks_like_section_start(query))


def section_heading_queries(queries: list[str] | None) -> list[str]:
    return [query for query in queries or [] if looks_like_section_start(query)]


def detail_forward_window_for_query(question: str, expanded_queries: list[str] | None = None) -> int:
    if not has_detail_intent(question):
        return 0
    if count_section_heading_queries(expanded_queries) > 1:
        if has_result_intent(question):
            return SINGLE_SECTION_DETAIL_FORWARD_WINDOW
        return MULTI_SECTION_DETAIL_FORWARD_WINDOW
    return SINGLE_SECTION_DETAIL_FORWARD_WINDOW


def retrieve_documents(
    question: str,
    retrieval_config: RetrievalConfig,
    expanded_queries: list[str] | None = None,
) -> list[Document]:
    db = load_vector_store()
    retriever = build_retriever(db, retrieval_config)
    queries = unique_queries(question, expanded_queries)
    vector_documents = []
    for query in queries:
        vector_documents.extend(retriever.invoke(query))

    if not retrieval_config.keyword_search and retrieval_config.neighbor_window <= 0:
        return vector_documents[: retrieval_config.max_context_docs]

    ordered_documents = get_ordered_documents(db)
    keyword_documents = []
    has_difficulty_targets = bool(extract_target_difficulty_numbers(question))
    if retrieval_config.keyword_search:
        difficulty_queries = expand_difficulty_queries(question, ordered_documents)
        difficulty_documents = keyword_search_each_query(
            difficulty_queries,
            ordered_documents,
            per_query_limit=2,
            total_limit=retrieval_config.max_context_docs,
        )
        heading_queries = [] if has_difficulty_targets else section_heading_queries(expanded_queries)
        section_documents = section_heading_search_documents(
            heading_queries,
            ordered_documents,
            total_limit=retrieval_config.max_context_docs,
        )
        keyword_documents = keyword_search_documents(
            "\n".join(queries),
            ordered_documents,
            retrieval_config.keyword_k,
        )
    else:
        difficulty_documents = []
        section_documents = []

    if section_documents:
        combined = dedupe_documents([*difficulty_documents, *section_documents])
    elif difficulty_documents:
        combined = dedupe_documents([*difficulty_documents, *keyword_documents])
    else:
        combined = dedupe_documents([*keyword_documents, *vector_documents])
    if has_detail_intent(question) and not has_list_intent(question) and not has_difficulty_targets:
        body_documents = [doc for doc in combined if not looks_like_catalog(doc.page_content)]
        if body_documents:
            combined = body_documents
    return expand_with_neighbors(
        combined,
        ordered_documents,
        retrieval_config.neighbor_window,
        retrieval_config.max_context_docs,
        detail_forward_window=detail_forward_window_for_query(question, expanded_queries),
    )
