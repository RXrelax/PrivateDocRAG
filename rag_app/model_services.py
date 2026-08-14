import streamlit as st
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek

from .env import get_dashscope_api_key, get_deepseek_api_key
from .config import INDEX_DIR
from .vector_store import get_index_version, index_operation_lock, validate_active_index


@st.cache_resource(show_spinner=False)
def get_cached_embeddings(dashscope_api_key: str) -> DashScopeEmbeddings:
    return DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=dashscope_api_key,
    )


def get_embeddings() -> DashScopeEmbeddings:
    return get_cached_embeddings(get_dashscope_api_key())


@st.cache_resource(show_spinner=False)
def load_cached_vector_store(
    index_version: tuple[int, ...],
    dashscope_api_key: str,
) -> FAISS:
    with index_operation_lock():
        validate_active_index()
        return FAISS.load_local(
            str(INDEX_DIR),
            get_cached_embeddings(dashscope_api_key),
            allow_dangerous_deserialization=True,
        )


def load_vector_store() -> FAISS:
    try:
        with index_operation_lock():
            validate_active_index()
            return load_cached_vector_store(
                get_index_version(),
                get_dashscope_api_key(),
            )
    except Exception as exc:
        raise RuntimeError("加载本地向量索引失败，请清除数据库后重新处理文档。") from exc


def clear_vector_store_cache() -> None:
    load_cached_vector_store.clear()


def get_llm() -> ChatDeepSeek:
    return ChatDeepSeek(
        model="deepseek-reasoner",
        api_key=get_deepseek_api_key(),
        streaming=False,
    )
