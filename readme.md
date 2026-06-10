# 文档智能助手

这是一个本地文档问答工程。主应用是 `langchain_rag.py`，用于上传 PDF 或 TXT 文件、构建本地 FAISS 向量索引，并基于 DeepSeek 与 DashScope Embeddings 进行检索问答。

## 功能概览

- 支持 PDF、TXT 文件上传。
- 使用 DashScope Embeddings 生成文本向量。
- 使用 FAISS 在本地保存向量索引。
- 支持 Similarity 和 MMR 两种检索模式。
- 支持多轮对话历史。
- 回答后只展示模型实际引用的来源片段。
- 处理文档时显示抽取、分片、向量化和保存进度。

## 快速启动

1. 创建并激活 Python 环境。

2. 安装依赖：

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

## 环境变量

| 变量名 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 调用 DeepSeek 聊天模型生成回答 |
| `DASHSCOPE_API_KEY` | 调用 DashScope Embeddings 生成向量 |
| `OPENWEATHER_API_KEY` | 课程示例中的 OpenWeather API 配置 |

主应用会优先读取大写环境变量。为了兼容旧配置，`DASHSCOPE_API_KEY` 也兼容旧的 `dashscope_api_key`。

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `langchain_rag.py` | 当前主应用 |
| `requirements.txt` | 主应用依赖 |
| `.env.example` | 环境变量模板，可以提交到仓库 |
| `.env` | 本机真实密钥文件，不应提交 |
| `work/` | 运行时索引、备份、调试日志和过程文件目录，不进入 git |
| `work/faiss_db/` | 应用运行时生成的 FAISS 索引目录 |
| `LangChain公开课/` | 课程资料和示例脚本 |
| `CHANGELOG.md` | 重要变更记录 |

## 使用建议

- 第一次使用时先上传文档并点击“处理文档”。
- 普通明确问题可以使用 Similarity。
- 需要覆盖多个角度或长文档综合问题时，优先使用 MMR。
- `k` 表示最终交给模型参考的片段数量。
- MMR 模式下，`fetch_k` 表示先取多少候选片段再做多样性筛选。

## 安全说明

- 不要提交 `.env`、日志、Notebook 输出或任何真实 API key。
- 如果 key 曾经出现在 git、Notebook 输出、日志、截图或同步盘中，应到对应服务控制台 revoke/删除旧 key，并重新生成新 key。
- 当前仓库不做 git history rewrite；历史中如果曾经提交过 key，仍应视为已泄漏。
- `weather_server.py` 已改为读取 `OPENWEATHER_API_KEY`，不再硬编码 OpenWeather key。

## 仓库状态待处理

- `dify` 当前在 git 中是 gitlink/submodule 形态，但仓库没有 `.gitmodules`，且工作区显示为删除状态。后续需要单独决定是补充 submodule 配置，还是从索引中移除。
- `langchain_rag.zip` 当前显示为删除状态。后续需要确认是否仍要保留压缩包。
