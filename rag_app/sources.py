import re

from langchain_core.documents import Document


def format_source_label(doc: Document) -> str:
    source = doc.metadata.get("source", "未知来源")
    page = doc.metadata.get("page")
    chunk_id = doc.metadata.get("chunk_id")
    parts = [str(source)]
    if page:
        parts.append(f"第 {page} 页")
    if chunk_id:
        parts.append(f"片段 {chunk_id}")
    return " / ".join(parts)


def extract_cited_source_numbers(answer: str, max_sources: int) -> list[int]:
    cited_numbers: list[int] = []
    seen = set()
    for match in re.finditer(r"(?:来源|source)\s*[:：#]?\s*(\d+)", answer, re.IGNORECASE):
        number = int(match.group(1))
        if 1 <= number <= max_sources and number not in seen:
            seen.add(number)
            cited_numbers.append(number)
    return cited_numbers


def preview_text(text: str, limit: int = 220) -> str:
    compact_text = re.sub(r"\s+", " ", text).strip()
    if len(compact_text) <= limit:
        return compact_text
    return compact_text[:limit].rstrip() + "..."


def build_source_entries(answer: str, source_documents: list[Document]) -> list[dict[str, str | int]]:
    if not source_documents:
        return []

    entries = []
    for number in extract_cited_source_numbers(answer, len(source_documents)):
        doc = source_documents[number - 1]
        entries.append(
            {
                "number": number,
                "label": format_source_label(doc),
                "preview": preview_text(doc.page_content),
            }
        )
    return entries


def build_retrieval_entries(source_documents: list[Document]) -> list[dict[str, str | int]]:
    entries = []
    for number, doc in enumerate(source_documents, start=1):
        entries.append(
            {
                "number": number,
                "label": format_source_label(doc),
                "preview": preview_text(doc.page_content),
                "chars": len(doc.page_content),
            }
        )
    return entries
