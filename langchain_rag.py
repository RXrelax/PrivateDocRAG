import streamlit as st

from rag_app.documents import DocumentProcessingError
from rag_app.env import get_missing_config, load_environment
from rag_app.indexing import IndexBuildError, process_documents
from rag_app.model_services import clear_vector_store_cache
from rag_app.qa import answer_question
from rag_app.sources import build_retrieval_entries, build_source_entries
from rag_app.ui import (
    ProcessingProgress,
    apply_chat_style,
    render_app_header,
    render_chat_history,
    render_empty_state,
    render_index_status,
    render_sidebar,
    reset_history,
    show_retrieval_trace,
    show_source_entries,
)
from rag_app.vector_store import clear_index_files, db_exists, read_index_manifest


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
        answer_style,
        uploaded_files,
        process_clicked,
        clear_clicked,
        reset_clicked,
    ) = render_sidebar(missing_config)

    if reset_clicked:
        reset_history()
        st.rerun()

    if clear_clicked:
        try:
            removed = clear_index_files()
            clear_vector_store_cache()
            reset_history()
            if removed:
                st.sidebar.success("已清除已知数据库文件、诊断日志并重置历史")
            else:
                st.sidebar.info("未找到可安全清除的已知数据库文件")
            st.rerun()
        except (OSError, RuntimeError):
            st.sidebar.error("清理数据库失败，请检查 work/ 目录权限或未知内容。")

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
                result = process_documents(
                    uploaded_files,
                    progress,
                    warn=st.sidebar.warning,
                )
                st.sidebar.write(f"原始文本长度：{result.raw_text_length} 字符")
                st.sidebar.write(f"分割得到 {result.chunk_count} 个文本片段")
                if result.chunk_count:
                    st.sidebar.success(f"向量索引已完成：{result.chunk_count} 个片段")
                    reset_history()
                    st.rerun()
                else:
                    st.sidebar.error("没有从上传的文件中抽取到任何文本，请检查文件内容是否可读。")
            except (DocumentProcessingError, IndexBuildError, RuntimeError) as exc:
                st.sidebar.error(f"处理文件失败：{exc}")

    db_ready = db_exists()
    if db_ready:
        st.sidebar.success("数据库状态：已就绪")
        render_index_status(read_index_manifest())
    else:
        st.sidebar.warning("请先上传并处理文件")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    render_chat_history()
    render_empty_state(db_ready, missing_config)

    user_question = st.chat_input(
        "询问这份文档...",
        disabled=not db_ready or bool(missing_config),
    )

    if not user_question:
        return

    st.session_state["history"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            try:
                answer, source_documents = answer_question(
                    user_question,
                    retrieval_config,
                    st.session_state.get("history", []),
                    answer_style=answer_style,
                )
                source_entries = build_source_entries(answer, source_documents)
                retrieval_entries = build_retrieval_entries(source_documents)
                st.markdown(answer)
                show_source_entries(source_entries)
                show_retrieval_trace(retrieval_entries, retrieval_config)
                st.session_state["history"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "cited_sources": source_entries,
                        "retrieved_sources": retrieval_entries,
                        "retrieval_config": retrieval_config.__dict__,
                    }
                )
            except RuntimeError as exc:
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
