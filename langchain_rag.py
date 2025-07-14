import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import TextLoader

# —— 加载环境变量 —— #
dotenv_path = os.getenv('DOTENV_PATH', None)
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv(override=True)

dashscope_api_key = os.getenv('DASHSCOPE_API_KEY')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# —— Embeddings & Prompt 模板 —— #
embeddings = DashScopeEmbeddings(
    model='text-embedding-v1',
    dashscope_api_key=dashscope_api_key
)
system_prompt = (
    "你是一个 AI 助手，只能从下方提供[object Object]上下文文本中获取信息来回答用户提问。请严格按照以下步骤执行："
    "理解问题：准确识别用户问题的核心。"
    "检索上下文：在提供的文本中查找所有相关内容"
    "构建回答："
    "如果上下文中能完整回答问题，则："
    "用明了但详细的语言输出答案。"
    "如果上下文中 确实没有 涉及问题所需信息，则只输出："
    "答案不在上下文中"
    "绝不输出任何脱离上下文的推测或信息。"
    "格式要求："
    "回答时不要添加与问题无关的介绍或总结；严格按照上面“能回答”／“答案不在上下文中”二选一；"
)
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('human', '上下文如下：\n{context}\n\n问题：{question}')
])

# —— 处理上传文档，并构建 FAISS 索引 —— #
def process_documents(uploaded_files):
    text = ""
    for file in uploaded_files:
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

    st.sidebar.write(f"🔍 原始文本长度：{len(text)} 字符")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    st.sidebar.write(f"📦 分割得到 {len(chunks)} 个文本片段")

    if not chunks:
        st.sidebar.error("🚨 没有从上传的文件中抽取到任何文本，请检查文件内容是否可读。")
        return 0

    store = FAISS.from_texts(chunks, embedding=embeddings)
    if os.path.isdir('faiss_db'):
        shutil.rmtree('faiss_db', ignore_errors=True)
    store.save_local('faiss_db')
    return len(chunks)

# —— 构建会话检索链 —— #
def get_chain():
    db = FAISS.load_local(
        'faiss_db',
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 800})

    streamlit_handler = StreamlitCallbackHandler(st.container())
    cb_manager = CallbackManager([streamlit_handler])

    llm = init_chat_model(
        'deepseek-reasoner',
        model_provider='deepseek',
        streaming=True,
        callback_manager=cb_manager
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    if 'history' in st.session_state:
        memory.chat_memory.messages = st.session_state['history']

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            'prompt': prompt_template,
            'document_variable_name': 'context'
        },
        verbose=True,
        return_source_documents=False
    )
    return chain

def db_exists():
    return os.path.isdir('faiss_db') and os.path.exists('faiss_db/index.faiss')

# —— 主函数 —— #
def main():
    st.set_page_config(
        page_title='文档智能助手',
        page_icon='📄',
        layout='wide'
    )
    st.title('📄 文档智能助手 v1.3.0')
    st.markdown(
        """
        欢迎使用文档智能助手！
        - 支持 PDF 与纯文本（.txt）文件的上传与智能问答
        - 实时检索与多轮对话记忆，采用 LangChain 官方接口
        - Streamlit 流式返回，让您直观地看到 LLM 推理过程
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
            st.sidebar.success(f'✅ 分割出 {count} 个文本片段，向量索引已完成')
            if 'history' in st.session_state:
                del st.session_state['history']
            st.rerun()

    if st.sidebar.button('🗑️ 清除数据库'):
        if db_exists():
            shutil.rmtree('faiss_db', ignore_errors=True)
            st.sidebar.success('🗑️ 数据库已清除，历史已重置')
            if 'history' in st.session_state:
                del st.session_state['history']
            st.rerun()
        else:
            st.sidebar.info('无需清除，数据库不存在')

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
                # 1. 检索
                docs = chain.retriever.get_relevant_documents(user_question)
                # 2. 检查是否已点击停止
                if st.session_state['stop_requested']:
                    st.warning("⚠️ 已终止思考，如需重新提问，请修改输入框内容。")
                else:
                    # 3. 生成回答
                    result = chain({"question": user_question})
                    answer = result['answer']
                    st.session_state['history'] = result['chat_history']
                    st.markdown(f'**{answer}**')
    else:
        st.write('请通过侧边栏上传并处理文件，然后开始提问。')

if __name__ == '__main__':
    main()
