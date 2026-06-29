# 文档智能助手

这是一个个人本地使用的文档问答工程。主入口仍然是 `langchain_rag.py`，用于上传 PDF 或 TXT 文件、构建本地 FAISS 向量索引，并基于 DeepSeek 与 DashScope Embeddings 进行检索问答。

## 功能概览

- 支持 PDF、TXT 文件上传。
- 使用 DashScope Embeddings 生成文本向量。
- 使用 FAISS 在 `work/faiss_db/` 保存本地索引。
- 支持 Similarity 和 MMR 两种检索模式。
- 支持多轮对话历史。
- 回答后只展示模型实际引用的来源片段。
- 处理文档时显示抽取、分片、向量化和保存进度。
- 索引生成后写入 `work/faiss_db/manifest.json`，记录创建时间、来源文件数、片段数和安全说明。

## 快速启动

1. 创建并激活 Python 环境。

2. 安装运行依赖：

```powershell
pip install -r requirements.txt
```

3. 复制 `.env.example` 为 `.env`，并填入本机真实 key：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

4. 启动应用：

```powershell
python -m streamlit run langchain_rag.py --server.address 127.0.0.1 --server.port 8501
```

5. 在浏览器打开：

```text
http://127.0.0.1:8501/
```

## 最小冒烟样例

可以先用一个很小的 TXT 文件验证流程，例如内容为：

```text
孙悟空曾被压在五行山下，后来随唐僧西行取经。
```

启动应用后上传该 TXT，点击“处理文档”，再提问“孙悟空后来做了什么？”。正常情况下，侧边栏会显示索引已就绪和 manifest 摘要，回答区会展示答案及引用来源。

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `langchain_rag.py` | Streamlit 入口，保持原启动方式 |
| `rag_app/` | 主应用内部模块：配置、文档处理、索引、检索问答、UI |
| `tests/` | 不触网的单元测试 |
| `requirements.txt` | 运行依赖 |
| `requirements-dev.txt` | 可选开发/测试依赖 |
| `.env.example` | 环境变量模板，可以提交到仓库 |
| `.env` | 本机真实密钥文件，不应提交 |
| `work/` | 运行时索引、调试日志和过程文件目录，不进入 git |
| `work/faiss_db/` | 应用运行时生成的 FAISS 索引目录 |
| `work/faiss_db/manifest.json` | 当前索引的来源与片段信息 |
| `LangChain公开课/` | 学习资料和课程示例，不作为主应用质量门槛 |
| `CHANGELOG.md` | 重要变更记录 |

## 环境变量

| 变量名 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 调用 DeepSeek 聊天模型生成回答 |
| `DASHSCOPE_API_KEY` | 调用 DashScope Embeddings 生成向量 |
| `OPENWEATHER_API_KEY` | 课程示例中的 OpenWeather API 配置 |

主应用会优先读取大写环境变量。为了兼容旧配置，`DASHSCOPE_API_KEY` 也兼容旧的 `dashscope_api_key`。

## 开发检查

安装可选开发依赖：

```powershell
pip install -r requirements-dev.txt
```

运行测试：

```powershell
python -m pytest
```

基础导入检查：

```powershell
python -c "import langchain_rag; print('IMPORT_OK')"
```

## 使用建议

- 第一次使用时先上传文档并点击“处理文档”。
- 普通明确问题可以使用 Similarity。
- 需要覆盖多个角度或长文档综合问题时，优先使用 MMR。
- `k` 表示最终交给模型参考的片段数量。
- MMR 模式下，`fetch_k` 表示先取多少候选片段再做多样性筛选。
- 如果侧边栏提示 manifest 缺失，建议清除数据库并重新处理文档。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 提示缺少 API key | 检查 `.env` 是否存在，变量名是否为 `DEEPSEEK_API_KEY` 和 `DASHSCOPE_API_KEY` |
| PDF 没有抽取出文本 | 可能是扫描版 PDF，需要先 OCR 成可复制文本 |
| 向量化失败 | 检查 DashScope key、网络、额度，或先用更小文档测试 |
| 生成回答失败 | 检查 DeepSeek key、网络、额度 |
| 加载本地索引失败 | 清除数据库后重新上传并处理文档 |

## 安全与数据说明

- 不要提交 `.env`、日志、Notebook 输出或任何真实 API key。
- 当前仓库不做 git history rewrite；历史中如果曾经提交过 key，仍应视为已泄漏，需要在对应服务控制台 revoke/删除旧 key 并重新生成。
- FAISS 会写入 `index.pkl`，主应用只从 `work/faiss_db/` 加载本地索引；如果索引不是本应用生成的，请清除并重新处理文档。
- 大 PDF、zip、提取后的 Markdown/图片等资料更适合放在外部资料目录或 Git LFS；仓库内建议只保留小样例和必要源码。
- 课程资料位于 `LangChain公开课/`，用于学习和参考，不建议把其中 Notebook 输出当作正式交付物。
