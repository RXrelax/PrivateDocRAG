import json
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_ERROR_LOG_PATH,
    INDEX_BACKUP_PREFIX,
    INDEX_DIR,
    INDEX_EMBEDDING_MODEL,
    INDEX_FILES,
    INDEX_MANIFEST_FILE,
    INDEX_SCHEMA_VERSION,
    INDEX_STAGING_PREFIX,
    INDEX_VECTOR_STORE,
    WORK_DIR,
)


class IndexStorageError(RuntimeError):
    pass


_INDEX_OPERATION_LOCK = threading.RLock()


@contextmanager
def index_operation_lock():
    with _INDEX_OPERATION_LOCK:
        yield


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _required_index_names() -> set[str]:
    return {*INDEX_FILES, INDEX_MANIFEST_FILE}


def ensure_work_dirs() -> Path:
    if WORK_DIR.is_symlink():
        raise IndexStorageError("运行目录不可信，请检查 work/ 配置。")
    if WORK_DIR.exists() and not WORK_DIR.is_dir():
        raise IndexStorageError("运行目录不可用，请检查 work/ 配置。")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if WORK_DIR.is_symlink() or not WORK_DIR.is_dir():
        raise IndexStorageError("运行目录不可信，请检查 work/ 配置。")
    return WORK_DIR.resolve()


def validate_index_manifest(manifest: object) -> dict:
    if not isinstance(manifest, dict):
        raise IndexStorageError("索引 manifest 格式无效。")
    if type(manifest.get("schema_version")) is not int:
        raise IndexStorageError("索引 manifest 与当前应用不兼容。")
    if (
        manifest.get("schema_version") != INDEX_SCHEMA_VERSION
        or manifest.get("vector_store") != INDEX_VECTOR_STORE
        or manifest.get("embedding_model") != INDEX_EMBEDDING_MODEL
    ):
        raise IndexStorageError("索引 manifest 与当前应用不兼容。")
    return manifest


def _read_manifest_file(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise IndexStorageError("索引 manifest 不是可信的普通文件。")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndexStorageError("索引 manifest 无法读取。") from exc
    return validate_index_manifest(manifest)


def validate_index_directory(index_dir: Path) -> dict:
    index_dir = Path(index_dir)
    if index_dir.is_symlink() or not index_dir.is_dir():
        raise IndexStorageError("索引目录不存在或不可信。")
    try:
        entries = list(index_dir.iterdir())
    except OSError as exc:
        raise IndexStorageError("索引目录无法读取。") from exc

    required_names = _required_index_names()
    if {entry.name for entry in entries} != required_names:
        raise IndexStorageError("索引目录缺少必要文件或包含未知内容。")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise IndexStorageError("索引文件不是可信的普通文件。")
    return _read_manifest_file(index_dir / INDEX_MANIFEST_FILE)


def validate_active_index() -> dict:
    if WORK_DIR.is_symlink() or not WORK_DIR.is_dir():
        raise IndexStorageError("运行目录不存在或不可信。")
    return validate_index_directory(INDEX_DIR)


def _assert_replaceable_active_directory() -> bool:
    if INDEX_DIR.is_symlink():
        raise IndexStorageError("现有索引目录不可信，拒绝替换。")
    if not INDEX_DIR.exists():
        return False
    if not INDEX_DIR.is_dir():
        raise IndexStorageError("现有索引路径不是目录，拒绝替换。")
    try:
        entries = list(INDEX_DIR.iterdir())
    except OSError as exc:
        raise IndexStorageError("现有索引目录无法检查，拒绝替换。") from exc

    required_names = _required_index_names()
    for entry in entries:
        if entry.name not in required_names:
            raise IndexStorageError("现有索引目录包含未知内容，拒绝替换。")
        if entry.is_symlink() or not entry.is_file():
            raise IndexStorageError("现有索引目录包含不可信内容，拒绝替换。")
    return True


def _owned_path(path: Path, prefix: str, work_root: Path) -> bool:
    return path.parent == work_root and path.name.startswith(prefix)


def _cleanup_owned_path(path: Path, prefix: str) -> None:
    work_root = ensure_work_dirs()
    path = Path(path)
    if not _owned_path(path, prefix, work_root):
        raise IndexStorageError("拒绝清理非本流程创建的路径。")
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def create_index_staging_dir() -> Path:
    work_root = ensure_work_dirs()
    return Path(tempfile.mkdtemp(prefix=INDEX_STAGING_PREFIX, dir=work_root))


def cleanup_index_staging_dir(staging_dir: Path) -> None:
    _cleanup_owned_path(Path(staging_dir), INDEX_STAGING_PREFIX)


def _new_backup_path(work_root: Path) -> Path:
    for _ in range(16):
        candidate = work_root / f"{INDEX_BACKUP_PREFIX}{uuid.uuid4().hex}"
        if not _path_exists(candidate):
            return candidate
    raise IndexStorageError("无法分配索引备份目录。")


def publish_staged_index(staging_dir: Path) -> bool:
    staging_dir = Path(staging_dir)
    with index_operation_lock():
        work_root = ensure_work_dirs()
        if not _owned_path(staging_dir, INDEX_STAGING_PREFIX, work_root):
            raise IndexStorageError("索引 staging 目录不可信。")
        validate_index_directory(staging_dir)
        had_active_index = _assert_replaceable_active_directory()
        backup_dir: Path | None = None

        if had_active_index:
            backup_dir = _new_backup_path(work_root)
            try:
                INDEX_DIR.rename(backup_dir)
            except OSError as exc:
                raise IndexStorageError("无法暂存旧索引，索引未更新。") from exc

        try:
            staging_dir.rename(INDEX_DIR)
        except OSError as switch_exc:
            if backup_dir is not None:
                try:
                    backup_dir.rename(INDEX_DIR)
                except OSError as restore_exc:
                    raise IndexStorageError("索引切换失败，且旧索引未能自动恢复。") from restore_exc
            raise IndexStorageError("索引切换失败，旧索引已保留。") from switch_exc

        try:
            validate_active_index()
        except Exception as validation_exc:
            try:
                INDEX_DIR.rename(staging_dir)
                if backup_dir is not None:
                    backup_dir.rename(INDEX_DIR)
            except OSError as restore_exc:
                raise IndexStorageError("新索引校验失败，且旧索引未能自动恢复。") from restore_exc
            if backup_dir is not None:
                raise IndexStorageError("新索引校验失败，旧索引已恢复。") from validation_exc
            raise IndexStorageError("新索引校验失败，索引未发布。") from validation_exc

        if backup_dir is not None:
            try:
                _cleanup_owned_path(backup_dir, INDEX_BACKUP_PREFIX)
            except (IndexStorageError, OSError):
                return False
        return True


def write_index_manifest(manifest: dict, index_dir: Path | None = None) -> None:
    validate_index_manifest(manifest)
    target_dir = Path(index_dir) if index_dir is not None else INDEX_DIR
    if index_dir is None and not _path_exists(target_dir):
        ensure_work_dirs()
        target_dir.mkdir()
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise IndexStorageError("无法写入不可信的索引目录。")

    target_path = target_dir / INDEX_MANIFEST_FILE
    if target_path.is_symlink() or (target_path.exists() and not target_path.is_file()):
        raise IndexStorageError("无法覆盖不可信的索引 manifest。")

    file_descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest-", dir=target_dir)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            json.dump(manifest, temporary_file, ensure_ascii=False, indent=2)
        temporary_path.replace(target_path)
    finally:
        if _path_exists(temporary_path):
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _unlink_known_file(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    return False


def clear_embedding_error_log() -> bool:
    if WORK_DIR.is_symlink() or (WORK_DIR.exists() and not WORK_DIR.is_dir()):
        return False
    return _unlink_known_file(EMBEDDING_ERROR_LOG_PATH)


def clear_index_files() -> bool:
    with index_operation_lock():
        removed = clear_embedding_error_log()
        if WORK_DIR.is_symlink() or not WORK_DIR.is_dir():
            return removed
        if INDEX_DIR.is_symlink() or not INDEX_DIR.is_dir():
            return removed

        for file_name in (*INDEX_FILES, INDEX_MANIFEST_FILE):
            removed = _unlink_known_file(INDEX_DIR / file_name) or removed
        try:
            INDEX_DIR.rmdir()
        except OSError:
            pass
        return removed


def db_exists() -> bool:
    with index_operation_lock():
        try:
            validate_active_index()
        except IndexStorageError:
            return False
        return True


def get_index_version() -> tuple[int, ...]:
    with index_operation_lock():
        validate_active_index()
        tracked_paths = [
            *(INDEX_DIR / file_name for file_name in INDEX_FILES),
            INDEX_DIR / INDEX_MANIFEST_FILE,
        ]
        version = []
        for path in tracked_paths:
            stat = path.stat()
            version.extend((stat.st_mtime_ns, stat.st_size))
        return tuple(version)


def _format_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def create_index_manifest(
    source_documents: list[Document],
    chunks: list[Document],
    raw_text_length: int,
) -> dict:
    sources: dict[tuple[str | None, str], dict] = {}
    for doc in source_documents:
        source = str(doc.metadata.get("source", "未知来源"))
        document_id = doc.metadata.get("document_id")
        document_key = (
            str(document_id) if document_id is not None else None,
            source,
        )
        entry = sources.setdefault(
            document_key,
            {
                "source": source,
                "document_id": document_key[0],
                "source_type": doc.metadata.get("source_type", "unknown"),
                "pages": set(),
                "document_units": 0,
                "text_length": 0,
            },
        )
        page = doc.metadata.get("page")
        if page:
            entry["pages"].add(page)
        entry["document_units"] += 1
        entry["text_length"] += len(doc.page_content)

    normalized_sources = []
    for entry in sources.values():
        pages = sorted(entry.pop("pages"))
        if pages:
            entry["page_count"] = len(pages)
            entry["first_page"] = pages[0]
            entry["last_page"] = pages[-1]
        normalized_sources.append(entry)

    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at": _format_utc_now(),
        "index_dir": str(INDEX_DIR),
        "vector_store": INDEX_VECTOR_STORE,
        "embedding_model": INDEX_EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "text_normalization": "NFKC",
        "chunk_count": len(chunks),
        "raw_text_length": raw_text_length,
        "source_count": len(normalized_sources),
        "sources": sorted(
            normalized_sources,
            key=lambda item: (item["source"], item["document_id"] or ""),
        ),
        "security_note": "本地 FAISS index.pkl 只应来自本应用生成的 work/faiss_db；来源不明时请清除并重新处理文档。",
    }


def read_index_manifest() -> dict | None:
    with index_operation_lock():
        try:
            return validate_active_index()
        except IndexStorageError:
            return None
