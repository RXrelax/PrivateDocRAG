# 变更记录

## 2026-06-10

本次改动围绕安全、可复现性、主应用稳定性、检索体验和界面交互进行整理。

### 安全与配置

- 新增 `.gitignore`，忽略 `.env`、`.env.*`、`work/`、`faiss_db/`、`__pycache__/`、Notebook checkpoint 等本地过程产物。
- 新增 `.env.example`，只保留环境变量占位符，不包含真实 key。
- 将 `.env` 从 git 跟踪中移除，但不删除本机文件。
- 将 `weather_server.py` 中的 OpenWeather key 改为读取 `OPENWEATHER_API_KEY`。
- 清理 Notebook 输出，降低 API key、运行日志和中间结果继续泄漏的风险。
- 未执行 git history rewrite；曾经暴露过的 DeepSeek、DashScope、OpenWeather key 仍需要在对应控制台手动 revoke 并重新生成。

### 主应用路径与索引

- 主应用统一使用 `WORK_DIR = Path("work")` 和 `INDEX_DIR = WORK_DIR / "faiss_db"`。
- 移除主应用中的递归删除逻辑。
- 清除索引时只删除明确的 `index.faiss` 和 `index.pkl`。
- 重新处理文档或清除索引后主动清理 FAISS 缓存，避免读取旧索引。
- TXT 文件改为直接解码上传内容，不再写系统临时目录。

### 文档处理与 metadata

- PDF 和 TXT 处理改为保留 `source`、`source_type`、`page`、`chunk_id` 等 metadata。
- 使用 `split_documents()` 保留文档来源信息。
- 构建 FAISS 索引时保留 metadata，便于回答后展示引用来源。
- 增加文档处理进度展示，覆盖文本抽取、文本分片、向量化、索引保存等阶段。
- 向量化改为分批调用，降低单次请求过大导致失败的概率，并在失败时显示批次和片段范围。

### 检索与问答

- 增加 Similarity 和 MMR 两种检索模式。
- 默认使用 MMR 多样召回。
- `k` 默认值调整为 8，并可在侧边栏配置。
- MMR 模式下可配置 `fetch_k` 和 `lambda_mult`，并保证 `fetch_k >= k`。
- 移除旧的重复检索调用，避免每次提问多做一次无用检索。
- 将旧的 `ConversationalRetrievalChain` 路线改为手动检索上下文后调用 `ChatDeepSeek`，规避 `langchain` 与 `langchain-core` 版本不兼容问题。
- 多轮历史改为使用 Streamlit session 中的普通消息列表，减少对旧 LangChain memory API 的依赖。

### UI 与交互

- 主界面改为更接近聊天助手的消息流布局。
- 使用 `st.chat_message` 展示历史消息。
- 使用 `st.chat_input` 作为底部输入框，避免旧 `text_input` 在页面重跑时重复触发同一个问题。
- 侧边栏保留文档上传、检索配置、清除数据库和重置对话。
- 停止按钮改为可靠的“重置对话”语义，不再承诺中断已经开始的同步 LLM 调用。
- Similarity 和 MMR 旁加入简短 tooltip。
- 回答不再整体加粗，保留模型返回的 Markdown。
- 参考来源改为“引用来源”：模型回答中标注 `[来源N]` 时，只展示实际被引用的来源；如果模型没有显式引用，则只兜底展示最靠前的一个命中片段，不再列出所有候选片段。

### 依赖与文档

- 新增 `requirements.txt`，记录主应用运行依赖。
- 重写 `readme.md`，保留快速启动、环境变量、目录说明、安全说明和使用建议。
- README 不再放版本号式历史摘要；详细改动统一记录在本文件。

### 已验证内容

- 使用 AST 解析 `langchain_rag.py`。
- 验证 `import langchain_rag` 正常。
- 验证 `ChatDeepSeek` 可初始化为 `deepseek-reasoner`。
- 验证 Similarity 与 MMR retriever 参数符合预期。
- 验证 Streamlit 临时端口可启动并返回 HTTP 200。
- 扫描旧接口残留，包括 `ConversationalRetrievalChain`、`ConversationBufferMemory`、`init_chat_model`、`langchain.callbacks` 和 `shutil.rmtree`。
- 扫描危险删除命令和真实 key 泄漏风险。

### 待处理

- `dify` 目前是缺失 `.gitmodules` 的 gitlink/submodule 状态，且工作区显示删除，需要后续单独决定保留方式。
- `langchain_rag.zip` 当前显示删除，需要后续确认是否仍要保留压缩包。
- 如果历史提交中曾包含真实 key，仍建议在外部控制台完成 key 旋转，并把 git history 清理作为单独任务处理。
