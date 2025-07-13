# 📦 v1.2.0 — 功能增强与优化汇总

### 一、架构与依赖管理

- **可配置的 dotenv 路径**  
  - 新增 `DOTENV_PATH` 环境变量，优先加载指定路径的 `.env` 文件，增强部署灵活性。

- **精简导入，移除冗余包**  
  - 去掉了 `langchain.tools.retriever`、`langchain.agents` 等多余依赖，改用更轻量的 `ConversationalRetrievalChain`。

### 二、PDF 处理流程优化

- **自动清理旧索引**  
  - 在每次重新处理时，自动删除已有的 `faiss_db` 目录，避免索引冲突或数据重复累积。

- **统一文本拼接与分割**  
  - 将 PDF 文本读取与分割合并到 `process_pdfs()`，返回分片数量并在侧边栏提示，简化流程。

### 三、检索与对话链改进

- **引入 ConversationalRetrievalChain**  
  - 用 LangChain 官方 `ConversationalRetrievalChain.from_llm` 替代自定义 Agent，减少手写逻辑、提高稳定性。

- **会话记忆（ConversationBufferMemory）**  
  - 支持多轮对话历史缓存，将上下文注入至 Memory，用户可连续提问并得到上下文关联的回答。

- **Streamlit 流式回调**  
  - 通过 `StreamlitCallbackHandler` + `CallbackManager`，实现 LLM 推理过程的实时流式输出，提升用户体验。

- **自定义系统模板**  
  - 将原 prompt 拆分为更严谨的「理解 → 检索 → 回答」三步流程，保证仅依据文档内容作答，不产生无据推测。

### 四、Session 管理与 UI 优化

- **Session State 持久化**  
  - 用 `st.session_state['history']` 存储对话历史，页面刷新（`st.rerun()`）后仍能保留上下文。

- **按钮权限与反馈**  
  - “Submit & Process” 与 “清除数据库” 按钮禁用/启用更合理；操作完成后自动刷新并给出成功/错误提醒。

- **页面配置与排版**  
  - 优化 `st.set_page_config`、标题、图标等信息；主界面与侧边栏布局更加清晰。

### 五、错误处理与健壮性

- **更完善的环境校验**  
  - 检查 `faiss_db/index.faiss` 文件存在性；不存在时在主界面提示“请先处理 PDF”。

- **容错加载**  
  - 加载 FAISS 时开启 `allow_dangerous_deserialization=True`，并捕获异常给出重试建议。

- **删除历史时同步清理 Memory**  
  - 清除数据库时，同时删除会话历史，防止 Memory 与索引不匹配导致错误。
