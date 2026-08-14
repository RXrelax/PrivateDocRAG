import json
import os
import tempfile
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS

from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_ERROR_LOG_PATH,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_BASE_SECONDS,
    INDEX_DIR,
    ProcessDocumentsResult,
)
from .documents import read_uploaded_documents, split_documents
from .model_services import clear_vector_store_cache, get_embeddings
from .vector_store import (
    IndexStorageError,
    cleanup_index_staging_dir,
    clear_embedding_error_log,
    create_index_manifest,
    create_index_staging_dir,
    ensure_work_dirs,
    publish_staged_index,
    write_index_manifest,
)


class IndexBuildError(RuntimeError):
    pass


def _chunk_diagnostics(chunks, start: int, end: int) -> list[dict]:
    diagnostics = []
    for offset, chunk in enumerate(chunks[start:end], start=start + 1):
        diagnostics.append(
            {
                "chunk_number": offset,
                "source": chunk.metadata.get("source"),
                "source_type": chunk.metadata.get("source_type"),
                "page": chunk.metadata.get("page"),
                "chunk_id": chunk.metadata.get("chunk_id"),
                "char_length": len(chunk.page_content),
            }
        )
    return diagnostics


def _write_embedding_error_log(
    chunks,
    start: int,
    end: int,
    exc: Exception,
    phase: str,
) -> None:
    payload = {
        "phase": phase,
        "range": {
            "start_chunk_number": start + 1,
            "end_chunk_number": end,
            "batch_size": end - start,
        },
        "error_type": exc.__class__.__name__,
        "error_message": "嵌入服务调用失败，原始异常未记录。",
        "chunks": _chunk_diagnostics(chunks, start, end),
        "note": "此文件不保存文档正文，只记录失败片段的来源、页码、chunk_id 和长度。",
    }
    parent = EMBEDDING_ERROR_LOG_PATH.parent
    temporary_path: Path | None = None
    try:
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            return
        parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".embedding-error-",
            dir=parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
        temporary_path.replace(EMBEDDING_ERROR_LOG_PATH)
    except OSError:
        return
    finally:
        if temporary_path and (temporary_path.exists() or temporary_path.is_symlink()):
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _embed_slice_with_retry(embeddings, texts: list[str], chunks, start: int, end: int) -> list:
    for attempt in range(EMBEDDING_MAX_RETRIES + 1):
        try:
            return embeddings.embed_documents(texts[start:end])
        except Exception as exc:
            if attempt < EMBEDDING_MAX_RETRIES:
                time.sleep(EMBEDDING_RETRY_BASE_SECONDS * (attempt + 1))
                continue
            if end - start == 1:
                _write_embedding_error_log(chunks, start, end, exc, phase="single_chunk")
                chunk = chunks[start]
                source = chunk.metadata.get("source", "未知来源")
                page = chunk.metadata.get("page")
                location = f"{source}"
                if page:
                    location += f" 第 {page} 页"
                raise IndexBuildError(
                    f"向量化片段 {start + 1} 失败（{location}，"
                    f"{len(chunk.page_content)} 字符）。诊断信息已写入 {EMBEDDING_ERROR_LOG_PATH}。"
                ) from exc

            midpoint = start + (end - start) // 2
            _write_embedding_error_log(chunks, start, end, exc, phase="split_batch")
            left_vectors = _embed_slice_with_retry(embeddings, texts, chunks, start, midpoint)
            right_vectors = _embed_slice_with_retry(embeddings, texts, chunks, midpoint, end)
            return [*left_vectors, *right_vectors]
    raise AssertionError("unreachable")


def build_vector_store(chunks, progress=None) -> FAISS:
    embeddings = get_embeddings()
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vectors = []
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for batch_index, start in enumerate(range(0, len(texts), EMBEDDING_BATCH_SIZE), start=1):
        end = min(start + EMBEDDING_BATCH_SIZE, len(texts))
        if progress:
            progress.update(
                72 + int((batch_index - 1) / max(total_batches, 1) * 16),
                "正在向量化文本",
                f"第 {batch_index}/{total_batches} 批，片段 {start + 1}-{end}",
            )
        batch_vectors = _embed_slice_with_retry(embeddings, texts, chunks, start, end)
        vectors.extend(batch_vectors)

    if progress:
        progress.update(88, "正在构建向量索引", "正在把向量写入 FAISS 内存索引")
    try:
        return FAISS.from_embeddings(
            zip(texts, vectors),
            embeddings,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise IndexBuildError("构建 FAISS 索引失败，请缩小文档规模后重试。") from exc


def _cleanup_staging_best_effort(staging_dir: Path) -> bool:
    try:
        cleanup_index_staging_dir(staging_dir)
    except (IndexStorageError, OSError):
        return False
    return True


def _save_vector_store_atomically(store, manifest: dict) -> bool:
    staging_dir = create_index_staging_dir()
    try:
        try:
            store.save_local(str(staging_dir))
        except Exception as exc:
            raise IndexBuildError("保存 FAISS 索引失败，请确认 work/ 目录可写。") from exc

        try:
            write_index_manifest(manifest, staging_dir)
        except Exception as exc:
            raise IndexBuildError("写入索引 manifest 失败，原索引保持不变。") from exc

        try:
            publication_clean = publish_staged_index(staging_dir)
        except IndexStorageError as exc:
            raise IndexBuildError(
                "发布新索引失败或未完全完成，请检查 work/ 目录后重试。"
            ) from exc
    except Exception:
        _cleanup_staging_best_effort(staging_dir)
        raise
    staging_clean = _cleanup_staging_best_effort(staging_dir)
    return publication_clean and staging_clean


def process_documents(uploaded_files, progress=None, warn=None) -> ProcessDocumentsResult:
    ensure_work_dirs()
    if progress:
        progress.update(3, "准备处理文件", f"共 {len(uploaded_files)} 个文件")

    source_documents = read_uploaded_documents(uploaded_files, progress, warn=warn)
    raw_text_length = sum(len(doc.page_content) for doc in source_documents)
    if progress:
        progress.update(55, "文本抽取完成", f"共抽取 {raw_text_length} 个字符")

    chunks = split_documents(source_documents)
    if progress:
        progress.update(65, "文本分片完成", f"共生成 {len(chunks)} 个文本片段")
    if not chunks:
        return ProcessDocumentsResult(chunk_count=0, raw_text_length=raw_text_length, manifest={})

    store = build_vector_store(chunks, progress)
    manifest = create_index_manifest(source_documents, chunks, raw_text_length)
    if progress:
        progress.update(90, "正在保存索引", f"索引将保存到 {INDEX_DIR}")
    publication_clean = _save_vector_store_atomically(store, manifest)
    if not publication_clean and warn:
        warn("新索引已发布，但旧备份或临时目录未能自动清理；请检查 work/。")
    try:
        clear_embedding_error_log()
    except OSError:
        if warn:
            warn("索引已发布，但旧的嵌入诊断日志未能清理。")
    try:
        clear_vector_store_cache()
    except Exception:
        if warn:
            warn("索引已发布，但内存缓存未能立即清理；后续加载会按索引版本刷新。")
    if progress:
        progress.complete(f"索引已保存，{len(chunks)} 个片段可用于问答")
    return ProcessDocumentsResult(
        chunk_count=len(chunks),
        raw_text_length=raw_text_length,
        manifest=manifest,
    )
