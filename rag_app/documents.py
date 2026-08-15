from collections.abc import Callable
import re
import unicodedata

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE


WarnCallback = Callable[[str], None]


class DocumentProcessingError(RuntimeError):
    pass


def decode_text_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def normalize_document_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"[\u200b-\u200f\ufeff]", "", normalized)
    return normalized


def decode_text_file(file) -> str:
    return normalize_document_text(decode_text_bytes(file.getvalue()))


def count_extraction_units(uploaded_files) -> int:
    units = 0
    for file in uploaded_files:
        if file.type == "application/pdf":
            try:
                reader = PdfReader(file)
                units += len(reader.pages)
                file.seek(0)
            except Exception:
                units += 1
        elif file.type == "text/plain":
            units += 1
    return max(units, 1)


def read_uploaded_documents(
    uploaded_files,
    progress=None,
    warn: WarnCallback | None = None,
) -> list[Document]:
    documents: list[Document] = []
    total_units = count_extraction_units(uploaded_files)
    completed_units = 0

    for upload_index, file in enumerate(uploaded_files, start=1):
        document_id = f"upload-{upload_index}"
        if file.type == "application/pdf":
            try:
                reader = PdfReader(file)
            except Exception as exc:
                raise DocumentProcessingError(
                    f"读取 PDF 失败：{file.name}。请确认文件未损坏，且不是只含图片的扫描件。"
                ) from exc

            total_pages = len(reader.pages)
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                except Exception as exc:
                    raise DocumentProcessingError(
                        f"抽取 PDF 文本失败：{file.name} 第 {page_number} 页。"
                    ) from exc
                if page_text:
                    page_text = normalize_document_text(page_text)
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file.name,
                                "source_type": "pdf",
                                "document_id": document_id,
                                "page": page_number,
                            },
                        )
                    )
                completed_units += 1
                if progress:
                    progress.update(
                        5 + int(completed_units / total_units * 45),
                        "正在抽取文本",
                        f"{file.name}：第 {page_number}/{total_pages} 页",
                    )
        elif file.type == "text/plain":
            text = decode_text_file(file)
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file.name,
                            "source_type": "txt",
                            "document_id": document_id,
                        },
                    )
                )
            completed_units += 1
            if progress:
                progress.update(
                    5 + int(completed_units / total_units * 45),
                    "正在抽取文本",
                    f"{file.name}：已读取文本文件",
                )
        else:
            if warn:
                warn(f"跳过不支持的文件类型：{file.name} ({file.type})")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    for chunk_id, chunk in enumerate(chunks, start=1):
        chunk.metadata = {**chunk.metadata, "chunk_id": chunk_id}
    return chunks
