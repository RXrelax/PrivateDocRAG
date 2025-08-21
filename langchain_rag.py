import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain.chains import GraphQAChain
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import TextLoader
from graphdatascience import GraphDataScience  # For community detection

# —— 加载环境变量 —— #
dotenv_path = os.getenv('DOTENV_PATH', None)
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv(override=True)
dashscope_api_key = os.getenv('DASHSCOPE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')  # 添加DeepSeek专用key，如果与DASHSCOPE不同
neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')  # 默认密码，实际应从env中安全加载
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# —— Embeddings & Prompt 模板 —— #
embeddings = DashScopeEmbeddings(
    model='text-embedding-v1',
    dashscope_api_key=dashscope_api_key
)
system_prompt = (
    """你是一个 AI 助手，主要任务是：在理解用户意图的基础上，从可用的上下文中获取信息，并灵活完成用户需求。请遵循以下指导原则：
理解问题：准确把握用户问题的核心和目的。
检索上下文：在提供的文本或资源中搜寻相关内容，优先使用上下文信息。
灵活应答：
如果上下文中包含完整且准确的信息，直接引用上下文，用清晰、详细的语言回答。
如果上下文不足以完全回答问题，可适度说明上下文限制，并在允许范围内补充必要的背景或合理推测。
如果用户的问题与上下文无关，不必强行检索上下文，直接按照用户指令完成任务。
避免无关内容：回答时聚焦用户需求，不要添加多余的介绍或总结。
明确反馈：若上下文确实无法提供关键答案，可提示“上下文有限，以下为基于已有信息的回答：”并继续提供最佳解答；仅在完全无法推断时，说明“信息不足，无法给出准确答案”。
注：以上原则请灵活应用，以用户满意为最终目标，不必过于死板地分步骤执行。"""
)
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('human', '上下文如下：\n{context}\n\n问题：{question}')
])

# —— 处理上传文档，并构建知识图谱 —— #
def process_documents(uploaded_files):
    text = ""
    for file in uploaded_files:
        st.sidebar.info(f"📄 处理文件: {file.name}")
        if file.type == "application/pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        elif file.type == "text/plain":
            # 写入临时文件以供 TextLoader 读取
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tf:
                tf.write(file.getbuffer())
                tmp_path = tf.name
            loader = TextLoader(tmp_path, encoding="utf-8")
            for page in loader.load():
                text += page.page_content
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        else:
            st.sidebar.warning(f"跳过不支持的文件类型：{file.name} ({file.type})")
    st.sidebar.info(f"🔍 原始文本长度：{len(text)} 字符")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    st.sidebar.info(f"📦 分割得到 {len(chunks)} 个文本片段")
    if not chunks:
        st.sidebar.error("🚨 没有从上传的文件中抽取到任何文本，请检查文件内容是否可读。")
        return 0

    # 初始化 LLM 和 GraphTransformer
    try:
        st.sidebar.info("🤖 初始化 LLM 和 GraphTransformer...")
        llm = init_chat_model('deepseek-reasoner', model_provider='deepseek', api_key=deepseek_api_key)  # 指定api_key如果需要
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["Person", "Organization", "Location", "Event", "Concept"],  # 扩展允许节点类型以适应通用文档
            allowed_relationships=["WORKS_AT", "LOCATED_IN", "PART_OF", "RELATED_TO", "HAS"],  # 扩展关系类型
            ignore_tool_usage=True  # 关键修改：禁用tool calling以兼容DeepSeek
            # 移除 node_properties 和 relationship_properties 以避免错误，因为DeepSeek不支持工具调用
        )
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        # 添加进度条以监控图谱转换
        progress = st.sidebar.progress(0)
        graph_docs = []
        for i, doc in enumerate(docs):
            st.sidebar.info(f"🔄 转换图谱文档 {i+1}/{len(docs)}...")
            graph_doc = transformer.convert_to_graph_documents([doc])  # 逐个处理以显示进度
            graph_docs.extend(graph_doc)
            progress.progress((i + 1) / len(docs))

        # 存储到 Neo4j
        st.sidebar.info("💾 连接 Neo4j 并存储图谱...")
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
        # 先清除旧图谱以避免冲突
        graph.query("MATCH (n) DETACH DELETE n")
        graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

        # 社区检测和总结（可选，提升全局搜索）
        st.sidebar.info("🧩 执行社区检测...")
        gds = GraphDataScience(neo4j_uri, auth=(neo4j_username, neo4j_password))
        G, _ = gds.graph.project("rag_graph", "Entity", "*")  # 用*匹配所有关系，避免特定名错误
        gds.leiden.write(G, writeProperty="community")
        gds.graph.drop(G)  # 清理项目

        st.sidebar.success(f'✅ 构建图谱：{len(graph_docs)} 个实体/关系')
        return len(graph_docs)
    except Exception as e:
        st.sidebar.error(f"🚨 图谱构建失败：{str(e)}。请检查 Neo4j 连接、LLM 兼容性或依赖安装。如果是DeepSeek相关错误，确认ignore_tool_usage=True已启用。")
        return 0

# —— 构建会话检索链（基于图谱） —— #
def get_chain():
    try:
        st.sidebar.info("🔗 加载图谱并构建检索链...")
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
        streamlit_handler = StreamlitCallbackHandler(st.container())
        cb_manager = CallbackManager([streamlit_handler])
        llm = init_chat_model(
            'deepseek-reasoner',
            model_provider='deepseek',
            streaming=True,
            callback_manager=cb_manager,
            api_key=deepseek_api_key  # 指定api_key
        )
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        if 'history' in st.session_state:
            memory.chat_memory.messages = st.session_state['history']
        chain = GraphQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            memory=memory,
            qa_prompt=prompt_template
        )
        st.sidebar.info("🔗 检索链构建完成")
        return chain
    except Exception as e:
        st.error(f"🚨 链构建失败：{str(e)}。请确保 Neo4j 已运行并正确配置。")
        return None

def db_exists():
    try:
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
        result = graph.query("MATCH (n) RETURN count(n) as count")
        return result[0]['count'] > 0
    except Exception:
        return False

# —— 主函数 —— #
def main():
    st.set_page_config(
        page_title='文档智能助手（GraphRAG 版）',
        page_icon='📄',
        layout='wide'
    )
    st.title('📄 文档智能助手 v1.3.0 (GraphRAG 集成)')
    st.markdown(
        """
        欢迎使用文档智能助手！现在集成 GraphRAG 以提升复杂查询准确性。
        - 支持 PDF 与纯文本（.txt）文件的上传与智能问答
        - 使用知识图谱检索，支持实体关系查询
        - 实时检索与多轮对话记忆，采用 LangChain 官方接口
        - Streamlit 流式返回，让您直观地看到 LLM 推理过程
        注意：确保 Neo4j 数据库运行，并设置环境变量 NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD。DeepSeek API 已兼容。
        """
    )
    # —— 侧边栏：文档管理 —— #
    st.sidebar.header('文档管理')
    uploaded_files = st.sidebar.file_uploader(
        '📎 上传 PDF 或 TXT 文件',
        accept_multiple_files=True,
        type=['pdf', 'txt']
    )
    if st.sidebar.button('🚀 Submit & Process', disabled=not uploaded_files):
        with st.spinner('📊 正在处理文件...'):
            count = process_documents(uploaded_files)
            st.sidebar.success(f'✅ 处理完成 {count} 个图谱元素')
            if 'history' in st.session_state:
                del st.session_state['history']
            st.rerun()
    if st.sidebar.button('🗑️ 清除数据库'):
        try:
            st.sidebar.info("🗑️ 清除图谱...")
            graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
            graph.query("MATCH (n) DETACH DELETE n")
            st.sidebar.success('🗑️ 图谱已清除，历史已重置')
            if 'history' in st.session_state:
                del st.session_state['history']
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"🚨 清除失败：{str(e)}")
    if db_exists():
        st.sidebar.success('✅ 数据库状态：已就绪')
    else:
        st.sidebar.warning('⚠️ 请先上传并处理文件')
    # —— 仅在数据库就绪后构建链 —— #
    if db_exists():
        chain = get_chain()
    else:
        chain = None
    # —— 初始化 Session State —— #
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'stop_requested' not in st.session_state:
        st.session_state['stop_requested'] = False
    # 输入变化时重置停止标志
    def reset_stop():
        st.session_state['stop_requested'] = False
    # —— 中心区域：问答框 —— #
    outer_left, outer_center, outer_right = st.columns([0.00001, 6, 0.00001])
    with outer_center:
        col_input, col_stop = st.columns([4, 1], gap="small")
        with col_input:
            user_question = st.text_input(
                '💬 请输入问题',
                placeholder='例如：这个文档的主要内容是什么？',
                key='user_input',
                on_change=reset_stop
            )
        with col_stop:
            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            if st.button('⏹️ 停止', key='stop_button', help="停止思考"):
                st.session_state['stop_requested'] = True
    # —— 问答逻辑 —— #
    if user_question:
        with st.spinner('🤔 AI 正在思考...'):
            if chain is None:
                st.error("请先在侧边栏上传并处理文件，再进行提问。")
            else:
                # 1. 检查是否已点击停止
                if st.session_state['stop_requested']:
                    st.warning("⚠️ 已终止思考，如需重新提问，请修改输入框内容。")
                else:
                    # 2. 生成回答
                    result = chain({"question": user_question})
                    answer = result['result']  # GraphQAChain 返回 'result'
                    st.session_state['history'] = result.get('chat_history', [])  # 更新历史
                    st.markdown(f'**{answer}**')
    else:
        st.write('请通过侧边栏上传并处理文件，然后开始提问。')

if __name__ == '__main__':
    main()