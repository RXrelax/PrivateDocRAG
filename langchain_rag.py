import os
import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter


WORK_DIR = Path("work")
INDEX_DIR = WORK_DIR / "faiss_db"
INDEX_FILES = ("index.faiss", "index.pkl")
DEFAULT_RETRIEVAL_K = 8
DEFAULT_FETCH_K = 40
DEFAULT_LAMBDA_MULT = 0.5
EMBEDDING_BATCH_SIZE = 25


@dataclass(frozen=True)
class RetrievalConfig:
    search_type: str
    k: int
    fetch_k: int = DEFAULT_FETCH_K
    lambda_mult: float = DEFAULT_LAMBDA_MULT


@dataclass
class ProcessingProgress:
    progress_bar: object
    status_text: object
    detail_text: object

    def update(self, percent: int, status: str, detail: str = "") -> None:
        safe_percent = max(0, min(100, percent))
        self.progress_bar.progress(safe_percent, text=status)
        self.status_text.markdown(f"**{status}**")
        if detail:
            self.detail_text.caption(detail)

    def complete(self, detail: str) -> None:
        self.update(100, "处理完成", detail)


def load_environment() -> None:
    dotenv_path = os.getenv("DOTENV_PATH")
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
    else:
        load_dotenv(override=True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def get_missing_config() -> list[str]:
    missing = []
    if not get_env("DASHSCOPE_API_KEY", "dashscope_api_key"):
        missing.append("DASHSCOPE_API_KEY")
    if not get_env("DEEPSEEK_API_KEY"):
        missing.append("DEEPSEEK_API_KEY")
    return missing


def ensure_work_dirs() -> None:
    WORK_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def clear_index_files() -> None:
    ensure_work_dirs()
    for file_name in INDEX_FILES:
        index_file = INDEX_DIR / file_name
        if index_file.exists():
            index_file.unlink()
    load_cached_vector_store.clear()


@st.cache_resource(show_spinner=False)
def get_cached_embeddings(dashscope_api_key: str) -> DashScopeEmbeddings:
    return DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=dashscope_api_key,
    )


def get_dashscope_api_key() -> str:
    dashscope_api_key = get_env("DASHSCOPE_API_KEY", "dashscope_api_key")
    if not dashscope_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法创建向量索引。")
    return dashscope_api_key


def get_embeddings() -> DashScopeEmbeddings:
    return get_cached_embeddings(get_dashscope_api_key())


def get_index_version() -> tuple[float, ...]:
    return tuple(
        (INDEX_DIR / file_name).stat().st_mtime if (INDEX_DIR / file_name).exists() else 0
        for file_name in INDEX_FILES
    )


@st.cache_resource(show_spinner=False)
def load_cached_vector_store(
    index_dir: str,
    index_version: tuple[float, ...],
    dashscope_api_key: str,
) -> FAISS:
    return FAISS.load_local(
        index_dir,
        get_cached_embeddings(dashscope_api_key),
        allow_dangerous_deserialization=True,
    )


def load_vector_store() -> FAISS:
    return load_cached_vector_store(
        str(INDEX_DIR),
        get_index_version(),
        get_dashscope_api_key(),
    )


def apply_chat_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 920px;
            padding-top: 2.4rem;
            padding-bottom: 7rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .app-header {
            margin: 0 auto 1.5rem auto;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .app-title-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .app-title {
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: 0;
        }

        .status-pill {
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.82rem;
            white-space: nowrap;
            color: rgba(255, 255, 255, 0.78);
        }

        .status-ready {
            border-color: rgba(54, 211, 153, 0.45);
            color: #9ee6c3;
        }

        .status-waiting {
            border-color: rgba(250, 204, 21, 0.35);
            color: #f7d774;
        }

        .app-subtitle {
            margin-top: 0.45rem;
            color: rgba(255, 255, 255, 0.62);
            font-size: 0.96rem;
        }

        div[data-testid="stChatMessage"] {
            padding: 1rem 0;
            background: transparent;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            font-size: 1rem;
            line-height: 1.75;
        }

        .empty-state {
            margin-top: 5rem;
            text-align: center;
            color: rgba(255, 255, 255, 0.74);
        }

        .empty-state h2 {
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
            letter-spacing: 0;
        }

        .empty-state p {
            margin: 0;
            color: rgba(255, 255, 255, 0.52);
        }

        .stChatInput {
            max-width: 920px;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(db_ready: bool, missing_config: list[str]) -> None:
    if missing_config:
        status_text = "配置待补齐"
        status_class = "status-waiting"
    elif db_ready:
        status_text = "数据库已就绪"
        status_class = "status-ready"
    else:
        status_text = "等待上传文档"
        status_class = "status-waiting"

    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title-row">
                <div class="app-title">文档智能助手</div>
                <div class="status-pill {status_class}">{status_text}</div>
            </div>
            <div class="app-subtitle">基于你的本地文档进行检索问答。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


system_prompt = """你是一个 AI 助手，主要任务是：在理解用户意图的基础上，从可用的上下文中获取信息，并灵活完成用户需求。请遵循以下指导原则：
理解问题：准确把握用户问题的核心和目的。
检索上下文：在提供的文本或资源中搜寻相关内容，优先使用上下文信息。
灵活应答：
如果上下文中包含完整且准确的信息，直接引用上下文，用清晰、详细的语言回答。
如果上下文不足以完全回答问题，可适度说明上下文限制，并在允许范围内补充必要的背景或合理推测。
如果用户的问题与上下文无关，不必强行检索上下文，直接按照用户指令完成任务。
避免无关内容：回答时聚焦用户需求，不要添加多余的介绍或总结。
明确反馈：若上下文确实无法提供关键答案，可提示“上下文有限，以下为基于已有信息的回答：”并继续提供最佳解答；仅在完全无法推断时，说明“信息不足，无法给出准确答案”。
注：以上原则请灵活应用，以用户满意为最终目标，不必过于死板地分步骤执行。"""

def decode_text_file(file) -> str:
    raw_bytes = file.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


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
    progress: ProcessingProgress | None = None,
) -> list[Document]:
    documents: list[Document] = []
    total_units = count_extraction_units(uploaded_files)
    completed_units = 0

    for file in uploaded_files:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            total_pages = len(reader.pages)
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file.name,
                                "source_type": "pdf",
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
            st.sidebar.warning(f"跳过不支持的文件类型：{file.name} ({file.type})")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    for chunk_id, chunk in enumerate(chunks, start=1):
        chunk.metadata = {**chunk.metadata, "chunk_id": chunk_id}
    return chunks


def build_vector_store(
    chunks: list[Document],
    progress: ProcessingProgress | None = None,
) -> FAISS:
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
        try:
            batch_vectors = embeddings.embed_documents(texts[start:end])
        except Exception as exc:
            raise RuntimeError(
                f"向量化第 {batch_index}/{total_batches} 批失败，"
                f"片段范围 {start + 1}-{end}：{exc}"
            ) from exc
        vectors.extend(batch_vectors)

    if progress:
        progress.update(88, "正在构建向量索引", "正在把向量写入 FAISS 内存索引")
    return FAISS.from_embeddings(
        zip(texts, vectors),
        embeddings,
        metadatas=metadatas,
    )


def process_documents(
    uploaded_files,
    progress: ProcessingProgress | None = None,
) -> int:
    ensure_work_dirs()
    if progress:
        progress.update(3, "准备处理文件", f"共 {len(uploaded_files)} 个文件")

    source_documents = read_uploaded_documents(uploaded_files, progress)
    raw_text_length = sum(len(doc.page_content) for doc in source_documents)
    st.sidebar.write(f"原始文本长度：{raw_text_length} 字符")
    if progress:
        progress.update(55, "文本抽取完成", f"共抽取 {raw_text_length} 个字符")

    chunks = split_documents(source_documents)
    st.sidebar.write(f"分割得到 {len(chunks)} 个文本片段")
    if progress:
        progress.update(65, "文本分片完成", f"共生成 {len(chunks)} 个文本片段")
    if not chunks:
        st.sidebar.error("没有从上传的文件中抽取到任何文本，请检查文件内容是否可读。")
        return 0

    store = build_vector_store(chunks, progress)
    if progress:
        progress.update(90, "正在保存索引", f"索引将保存到 {INDEX_DIR}")
    clear_index_files()
    store.save_local(str(INDEX_DIR))
    load_cached_vector_store.clear()
    if progress:
        progress.complete(f"索引已保存，{len(chunks)} 个片段可用于问答")
    return len(chunks)


def build_retriever(db: FAISS, config: RetrievalConfig):
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


def get_deepseek_api_key() -> str:
    deepseek_api_key = get_env("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，无法生成回答。")
    return deepseek_api_key


def get_llm() -> ChatDeepSeek:
    return ChatDeepSeek(
        model="deepseek-reasoner",
        api_key=get_deepseek_api_key(),
        streaming=False,
    )


def retrieve_documents(question: str, retrieval_config: RetrievalConfig) -> list[Document]:
    db = load_vector_store()
    retriever = build_retriever(db, retrieval_config)
    return retriever.invoke(question)


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


def answer_question(question: str, retrieval_config: RetrievalConfig) -> tuple[str, list[Document]]:
    source_documents = retrieve_documents(question, retrieval_config)
    context = build_context(source_documents)
    history_text = build_history_text(st.session_state.get("history", []))
    user_prompt = (
        "请结合检索上下文回答用户问题。若上下文不足，请明确说明限制并给出最佳回答。\n"
        "引用规则：只有当某条上下文直接支撑你的说法时，才在相关句子后标注 [来源N]；"
        "不要引用未使用的候选来源，也不要把所有候选来源都列出来。\n\n"
        f"对话历史：\n{history_text}\n\n"
        f"检索上下文：\n{context}\n\n"
        f"用户问题：{question}"
    )
    response = get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return normalize_message_content(response.content), source_documents


def db_exists() -> bool:
    return INDEX_DIR.is_dir() and all((INDEX_DIR / file_name).exists() for file_name in INDEX_FILES)


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

    cited_numbers = extract_cited_source_numbers(answer, len(source_documents))
    if not cited_numbers:
        cited_numbers = [1]

    entries = []
    for number in cited_numbers:
        doc = source_documents[number - 1]
        entries.append(
            {
                "number": number,
                "label": format_source_label(doc),
                "preview": preview_text(doc.page_content),
            }
        )
    return entries


def show_source_entries(source_entries: list[dict[str, str | int]]) -> None:
    if not source_entries:
        return

    with st.expander("引用来源", expanded=False):
        for entry in source_entries:
            st.markdown(f"**来源 {entry['number']}** · {entry['label']}")
            if entry["preview"]:
                st.caption(entry["preview"])


def reset_history() -> None:
    if "history" in st.session_state:
        del st.session_state["history"]


def reset_chat() -> None:
    reset_history()


def render_chat_history() -> None:
    for message in st.session_state.get("history", []):
        role = message.get("role", "assistant") if isinstance(message, dict) else "assistant"
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                show_source_entries(message.get("sources", []))


def render_empty_state(db_ready: bool, missing_config: list[str]) -> None:
    if st.session_state.get("history"):
        return

    if missing_config:
        title = "先补齐配置"
        body = "需要 API Key 后才能处理文档和生成回答。"
    elif not db_ready:
        title = "上传文档后开始提问"
        body = "处理完成后，我会根据检索片段回答你的问题。"
    else:
        title = "想问这份文档什么？"
        body = "可以直接输入问题，我会只展示回答实际引用的来源。"

    st.markdown(
        f"""
        <div class="empty-state">
            <h2>{title}</h2>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_retrieval_tooltips() -> None:
    st.markdown(
        """
        <span title="按问题相似度返回最接近的片段，适合目标明确的查询。"
              style="border-bottom: 1px dotted currentColor; cursor: help;">Similarity ?</span>
        &nbsp;&nbsp;
        <span title="兼顾相关性和结果多样性，适合需要覆盖多个角度的查询。"
              style="border-bottom: 1px dotted currentColor; cursor: help;">MMR ?</span>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(missing_config: list[str]) -> tuple[RetrievalConfig, list, bool, bool, bool]:
    st.sidebar.header("文档管理")

    with st.sidebar.expander("高级检索设置", expanded=True):
        render_retrieval_tooltips()
        mode = st.radio(
            "检索模式",
            options=["MMR 多样召回", "Similarity 精准匹配"],
            index=0,
        )
        retrieval_k = st.slider(
            "返回片段数 k",
            min_value=1,
            max_value=20,
            value=DEFAULT_RETRIEVAL_K,
            help="最终交给模型参考的片段数量。",
        )

        if mode.startswith("MMR"):
            fetch_k = st.slider(
                "候选池 fetch_k",
                min_value=retrieval_k,
                max_value=80,
                value=max(DEFAULT_FETCH_K, retrieval_k),
                help="MMR 先查看的候选片段数量，会自动不小于 k。",
            )
            lambda_mult = st.slider(
                "多样性 lambda_mult",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_LAMBDA_MULT,
                step=0.05,
                help="越接近 1 越偏相关性，越接近 0 越偏多样性。",
            )
            retrieval_config = RetrievalConfig(
                search_type="mmr",
                k=retrieval_k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
        else:
            retrieval_config = RetrievalConfig(
                search_type="similarity",
                k=retrieval_k,
            )

    uploaded_files = st.sidebar.file_uploader(
        "上传 PDF 或 TXT 文件",
        accept_multiple_files=True,
        type=["pdf", "txt"],
    )
    process_clicked = st.sidebar.button(
        "处理文档",
        disabled=not uploaded_files or bool(missing_config),
    )
    clear_clicked = st.sidebar.button("清除数据库")
    reset_clicked = st.sidebar.button("重置对话")
    return retrieval_config, uploaded_files, process_clicked, clear_clicked, reset_clicked


def main() -> None:
    load_environment()
    st.set_page_config(
        page_title="文档智能助手",
        page_icon="📄",
        layout="wide",
    )
    apply_chat_style()

    missing_config = get_missing_config()
    (
        retrieval_config,
        uploaded_files,
        process_clicked,
        clear_clicked,
        reset_clicked,
    ) = render_sidebar(missing_config)

    if reset_clicked:
        reset_history()
        st.rerun()

    if clear_clicked:
        if db_exists():
            clear_index_files()
            st.sidebar.success("数据库已清除，历史已重置")
            reset_history()
            st.rerun()
        else:
            st.sidebar.info("无需清除，数据库不存在")

    render_app_header(db_exists(), missing_config)

    if missing_config:
        st.warning(
            "缺少必要环境变量："
            + "、".join(missing_config)
            + "。请参考 .env.example 配置本机 .env。"
        )

    if process_clicked:
        progress_panel = st.container()
        with progress_panel:
            st.subheader("文件处理进度")
            progress = ProcessingProgress(
                progress_bar=st.progress(0, text="准备处理文件"),
                status_text=st.empty(),
                detail_text=st.empty(),
            )
        with st.spinner("正在处理文件，请保持页面打开..."):
            try:
                count = process_documents(uploaded_files, progress)
                if count:
                    st.sidebar.success(f"分割出 {count} 个文本片段，向量索引已完成")
                    reset_history()
                    st.rerun()
            except Exception as exc:
                st.sidebar.error(f"处理文件失败：{exc}")

    db_ready = db_exists()
    if db_ready:
        st.sidebar.success("数据库状态：已就绪")
    else:
        st.sidebar.warning("请先上传并处理文件")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    render_chat_history()
    render_empty_state(db_ready, missing_config)

    input_disabled = not db_ready or bool(missing_config)
    user_question = st.chat_input(
        "询问这份文档...",
        disabled=input_disabled,
    )

    if not user_question:
        return

    st.session_state["history"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            try:
                answer, source_documents = answer_question(user_question, retrieval_config)
                source_entries = build_source_entries(answer, source_documents)
                st.markdown(answer)
                show_source_entries(source_entries)
                st.session_state["history"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_entries,
                    }
                )
            except Exception as exc:
                error_message = f"生成回答失败：{exc}"
                st.error(error_message)
                st.session_state["history"].append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


if __name__ == "__main__":
    main()
