import math
import re
import unicodedata

from langchain_core.documents import Document

from .config import RetrievalConfig
from .model_services import load_vector_store


class RetrievalError(RuntimeError):
    pass


LOW_SIGNAL_TERMS = {
    "什么",
    "一下",
    "讲一",
    "讲下",
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
    "讲一下",
    "讲讲",
    "怎么",
    "如何",
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


def normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[\u200b-\u200f\ufeff]", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


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


def split_meaningful_runs(run: str) -> list[str]:
    splitter = "|".join(re.escape(token) for token in TERM_SPLIT_TOKENS)
    return [part for part in re.split(splitter, run) if len(part) >= 2]


def extract_container_terms(question: str) -> set[str]:
    normalized = normalize_search_text(question)
    terms: set[str] = set()
    for match in re.finditer(r"(\d{1,3})\s*难\s*(?:中|里|当中|里面)", normalized):
        number = int(match.group(1))
        terms.add(f"{number}难")
        chinese_number = number_to_chinese_under_100(number)
        if chinese_number:
            terms.add(f"{chinese_number}难")
    for match in re.finditer(
        r"([一二三四五六七八九十]{1,3})难\s*(?:中|里|当中|里面)",
        normalized,
    ):
        terms.add(f"{match.group(1)}难")
    return terms


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
            for width in range(2, min(6, len(part)) + 1):
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
        doc.metadata.get("document_id"),
        doc.metadata.get("source"),
        doc.metadata.get("chunk_id"),
        doc.metadata.get("page"),
        doc.page_content[:80],
    )


def is_low_signal_term(term: str) -> bool:
    if term in LOW_SIGNAL_TERMS:
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


def looks_like_catalog(text: str) -> bool:
    normalized = normalize_search_text(text)
    heading_count = len(
        re.findall(r"第?[一二三四五六七八九十百千万0-9]{1,5}[回章节难]", normalized)
    )
    return heading_count >= 8 or ("目录" in normalized[:120] and heading_count >= 3)


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
    keyword_weights = build_keyword_weights(
        terms,
        normalized_documents,
        downweighted_terms=extract_container_terms(question),
    )
    if not keyword_weights:
        return []

    wants_detail = has_detail_intent(question)
    wants_list = has_list_intent(question)

    scored = []
    for position, (doc, text) in enumerate(zip(documents, normalized_documents)):
        score = 0.0
        matched_terms = 0
        for term, weight in keyword_weights.items():
            count = text.count(term)
            if count:
                score += count * weight
                matched_terms += 1
        if score:
            score *= 1 + min(matched_terms / 5, 1.0)
            if wants_detail and not wants_list and looks_like_catalog(text):
                score *= 0.45
            scored.append((score, position, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _score, _position, doc in scored[:limit]]


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


def document_scope_key(doc: Document) -> tuple[str, object] | None:
    document_id = doc.metadata.get("document_id")
    if document_id not in (None, ""):
        return ("document_id", document_id)

    source = doc.metadata.get("source")
    if source in (None, ""):
        return None
    return ("source", source)


def expand_with_neighbors(
    documents: list[Document],
    ordered_documents: list[Document],
    window: int,
    limit: int,
) -> list[Document]:
    if limit <= 0:
        return []

    seed_documents = dedupe_documents(documents)[:limit]
    if window <= 0 or not seed_documents or len(seed_documents) >= limit:
        return seed_documents

    positions = {document_key(doc): index for index, doc in enumerate(ordered_documents)}
    expanded = list(seed_documents)
    seen = {document_key(doc) for doc in expanded}
    for doc in seed_documents:
        position = positions.get(document_key(doc))
        scope_key = document_scope_key(doc)
        if position is None or scope_key is None:
            continue

        for distance in range(1, window + 1):
            for neighbor_position in (position - distance, position + distance):
                if not 0 <= neighbor_position < len(ordered_documents):
                    continue
                neighbor = ordered_documents[neighbor_position]
                neighbor_key = document_key(neighbor)
                if neighbor_key in seen or document_scope_key(neighbor) != scope_key:
                    continue
                seen.add(neighbor_key)
                expanded.append(neighbor)
                if len(expanded) >= limit:
                    return expanded
    return expanded


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


def _retrieve_documents(
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
        return dedupe_documents(vector_documents)[: retrieval_config.max_context_docs]

    ordered_documents = get_ordered_documents(db)
    keyword_documents = []
    if retrieval_config.keyword_search:
        keyword_documents = keyword_search_documents(
            "\n".join(queries),
            ordered_documents,
            retrieval_config.keyword_k,
        )

    combined = dedupe_documents([*keyword_documents, *vector_documents])
    return expand_with_neighbors(
        combined,
        ordered_documents,
        retrieval_config.neighbor_window,
        retrieval_config.max_context_docs,
    )


def retrieve_documents(
    question: str,
    retrieval_config: RetrievalConfig,
    expanded_queries: list[str] | None = None,
) -> list[Document]:
    try:
        return _retrieve_documents(question, retrieval_config, expanded_queries)
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError("检索文档失败，请确认索引完整且外部服务可用。") from exc
