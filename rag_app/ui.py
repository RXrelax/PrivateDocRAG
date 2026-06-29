import streamlit as st

from .config import (
    DEFAULT_FETCH_K,
    DEFAULT_KEYWORD_K,
    DEFAULT_LAMBDA_MULT,
    DEFAULT_MAX_CONTEXT_DOCS,
    DEFAULT_NEIGHBOR_WINDOW,
    DEFAULT_QUERY_EXPANSION,
    DEFAULT_RETRIEVAL_K,
    RetrievalConfig,
)


class ProcessingProgress:
    def __init__(self, progress_bar, status_text, detail_text) -> None:
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.detail_text = detail_text

    def update(self, percent: int, status: str, detail: str = "") -> None:
        safe_percent = max(0, min(100, percent))
        self.progress_bar.progress(safe_percent, text=status)
        self.status_text.markdown(f"**{status}**")
        if detail:
            self.detail_text.caption(detail)

    def complete(self, detail: str) -> None:
        self.update(100, "处理完成", detail)


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


def render_sidebar(missing_config: list[str]) -> tuple[RetrievalConfig, str, list, bool, bool, bool]:
    st.sidebar.header("文档管理")

    with st.sidebar.expander("高级检索设置", expanded=True):
        answer_style = st.selectbox(
            "回答详略",
            options=["详细解释", "标准回答", "简短回答"],
            index=0,
            help="控制模型回答的展开程度。详细解释会更适合剧情梳理、章节总结和原因分析。",
        )
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
        keyword_search = st.checkbox(
            "关键词补召回",
            value=True,
            help="结合本地关键词匹配，改善编号、专名、章节名等精确查询。",
        )
        query_expansion = st.checkbox(
            "智能改写查询",
            value=DEFAULT_QUERY_EXPANSION,
            help="先用模型生成少量同义/别名/编号改写，再一起检索。会多调用一次聊天模型。",
        )
        neighbor_window = st.slider(
            "相邻片段补充",
            min_value=0,
            max_value=2,
            value=DEFAULT_NEIGHBOR_WINDOW,
            help="命中某个片段时，额外补充前后片段，减少上下文断裂。",
        )
        max_context_docs = st.slider(
            "最多送入片段",
            min_value=retrieval_k,
            max_value=24,
            value=max(DEFAULT_MAX_CONTEXT_DOCS, retrieval_k),
            help="向量、关键词和相邻片段合并后，最多交给模型的片段数量。",
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
                keyword_search=keyword_search,
                keyword_k=DEFAULT_KEYWORD_K,
                neighbor_window=neighbor_window,
                max_context_docs=max_context_docs,
                query_expansion=query_expansion,
            )
        else:
            retrieval_config = RetrievalConfig(
                search_type="similarity",
                k=retrieval_k,
                keyword_search=keyword_search,
                keyword_k=DEFAULT_KEYWORD_K,
                neighbor_window=neighbor_window,
                max_context_docs=max_context_docs,
                query_expansion=query_expansion,
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
    return retrieval_config, answer_style, uploaded_files, process_clicked, clear_clicked, reset_clicked


def render_index_status(manifest: dict | None) -> None:
    if not manifest:
        st.sidebar.caption("索引状态：未找到 manifest，建议重新处理文档。")
        return

    created_at = manifest.get("created_at", "未知时间")
    source_count = manifest.get("source_count", 0)
    chunk_count = manifest.get("chunk_count", 0)
    st.sidebar.caption(f"索引创建：{created_at}")
    st.sidebar.caption(f"来源：{source_count} 个文件，{chunk_count} 个片段")


def show_source_entries(source_entries: list[dict[str, str | int]]) -> None:
    if not source_entries:
        return

    with st.expander(f"实际引用来源（{len(source_entries)}）", expanded=False):
        for entry in source_entries:
            st.markdown(f"**来源 {entry['number']}** · {entry['label']}")
            if entry["preview"]:
                st.caption(entry["preview"])


def show_retrieval_trace(
    retrieval_entries: list[dict[str, str | int]],
    retrieval_config: RetrievalConfig,
) -> None:
    if not retrieval_entries:
        return

    mode = "MMR 多样召回" if retrieval_config.search_type == "mmr" else "Similarity 精准匹配"
    settings = f"模式：{mode}；k={retrieval_config.k}"
    if retrieval_config.search_type == "mmr":
        settings += f"；fetch_k={retrieval_config.fetch_k}；lambda_mult={retrieval_config.lambda_mult:.2f}"
    if retrieval_config.keyword_search:
        settings += f"；关键词补召回={retrieval_config.keyword_k}"
    if retrieval_config.query_expansion:
        settings += "；智能改写=开"
    if retrieval_config.neighbor_window:
        settings += f"；相邻片段=±{retrieval_config.neighbor_window}"

    with st.expander(f"检索过程（{len(retrieval_entries)} 个候选片段）", expanded=False):
        st.caption(settings)
        for entry in retrieval_entries:
            chars = entry.get("chars")
            char_text = f" · {chars} 字符" if chars else ""
            st.markdown(f"**候选 {entry['number']}** · {entry['label']}{char_text}")
            if entry["preview"]:
                st.caption(entry["preview"])


def reset_history() -> None:
    if "history" in st.session_state:
        del st.session_state["history"]


def render_chat_history() -> None:
    for message in st.session_state.get("history", []):
        role = message.get("role", "assistant") if isinstance(message, dict) else "assistant"
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                show_source_entries(message.get("cited_sources", message.get("sources", [])))
                retrieval_config = message.get("retrieval_config")
                if retrieval_config:
                    show_retrieval_trace(
                        message.get("retrieved_sources", []),
                        RetrievalConfig(**retrieval_config),
                    )


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
