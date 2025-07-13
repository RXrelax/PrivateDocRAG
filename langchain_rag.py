import os
import shutil
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

# 加载环境变量
dotenv_path = os.getenv('DOTENV_PATH', None)
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv(override=True)

# 从环境变量获取 API Key
dashscope_api_key = os.getenv('DASHSCOPE_API_KEY')

# 允许 KMP 重复加载以避免冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 初始化 Embeddings
embeddings = DashScopeEmbeddings(
    model='text-embedding-v1',
    dashscope_api_key=dashscope_api_key
)

# 系统提示词模板
system_prompt = (
    "你是一个 AI 助手，只能从下方提供的上下文文本中获取信息来回答用户提问。请严格按照以下步骤执行："
    "理解问题：准确识别用户问题的核心。"
    "检索上下文：在提供的文本中查找所有相关内容，记录对应位置（如段落号、行号或关键词）。"
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

def process_pdfs(pdf_files):
    """
    读取上传的 PDF 文件，抽取文本，分割成片段，构建 FAISS 索引。
    返回分割出的文本片段数量。
    """
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    store = FAISS.from_texts(chunks, embedding=embeddings)
    # 删除旧索引
    if os.path.isdir('faiss_db'):
        shutil.rmtree('faiss_db', ignore_errors=True)
    store.save_local('faiss_db')

    return len(chunks)

def get_chain():
    """
    加载本地 FAISS 索引，创建 Retriever、LLM、Memory，并组装成 ConversationalRetrievalChain。
    """
    db = FAISS.load_local(
        'faiss_db',
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever()

    # Streamlit 实时流式回调
    streamlit_handler = StreamlitCallbackHandler(st.container())
    cb_manager = CallbackManager([streamlit_handler])

    # 初始化聊天模型（开启流式输出）
    llm = init_chat_model(
        'deepseek-reasoner',
        model_provider='deepseek',
        streaming=True,
        callback_manager=cb_manager
    )

    # 对话记忆
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    # 如果已有历史，就注入
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
    """
    检查本地 FAISS 索引是否已存在。
    """
    return os.path.isdir('faiss_db') and os.path.exists('faiss_db/index.faiss')

def main():
    st.set_page_config(page_title='PDF RAG Chatbot', page_icon='🤖')
    st.title('PDF RAG AI 询问系统')

    # 侧边栏：文档管理
    st.sidebar.header('文档管理')
    pdf_files = st.sidebar.file_uploader(
        '📎 上传 PDF 文件',
        accept_multiple_files=True,
        type=['pdf']
    )

    # 处理上传按钮
    if st.sidebar.button('🚀 Submit & Process', disabled=not pdf_files):
        with st.spinner('📊 正在处理 PDF...'):
            count = process_pdfs(pdf_files)
            st.sidebar.success(f'✅ 分割出 {count} 个文本片段，向量索引已完成')
            # 清空历史，强制重跑
            if 'history' in st.session_state:
                del st.session_state['history']
            st.rerun()

    # 清除数据库按钮
    if st.sidebar.button('🗑️ 清除数据库'):
        if db_exists():
            shutil.rmtree('faiss_db', ignore_errors=True)
            st.sidebar.success('🗑️ 数据库已清除')
            if 'history' in st.session_state:
                del st.session_state['history']
            st.rerun()
        else:
            st.sidebar.info('无需清除，数据库不存在')

    # 索引状态提示
    if db_exists():
        st.sidebar.success('✅ 数据库状态：已就绪')
    else:
        st.sidebar.warning('⚠️ 请先上传并处理 PDF 文件')

    # 主界面：提问聊天
    if db_exists():
        chain = get_chain()
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        user_question = st.text_input(
            '💬 请输入问题',
            placeholder='例如：这个文档的主要内容是什么？'
        )

        if user_question:
            with st.spinner('🤔 AI 正在思考...'):
                result = chain({'question': user_question})
                answer = result['answer']
                st.session_state['history'] = result['chat_history']
                st.markdown(f'**🤖 AI:** {answer}')
    else:
        st.write('请通过侧边栏上传并处理 PDF，然后开始提问。')

if __name__ == '__main__':
    main()
