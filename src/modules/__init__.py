"""modules 层：core 接口的具体实现。

- llm         ：OpenAI 兼容 LLM 客户端（显式超时 + 退避重试）
- embedder    ：本地向量化（fastembed / BGE，ONNX 运行时）
- ingest      ：文档加载与语义切分（.md / .txt / .pdf）
- vectorstore ：向量索引（FAISS flat/hnsw + NumPy 兜底）与后端工厂
- store       ：索引构建、持久化、元信息与新鲜度校验（含 BM25 语料）
- retriever   ：混合检索（BM25 + 向量 + RRF 融合，支持多路加权）
- reranker    ：重排（Cross-Encoder / LLM / none，含降级链路）
- rewrite     ：查询改写（LLM 指代消解 + 扩展 / 规则拼接 / 关闭）
- memory      ：多轮记忆（滑动窗口 + 摘要压缩 + 会话仓库与 TTL）

这一层允许依赖第三方 SDK（openai / fastembed / faiss / jieba / rank_bm25）。
"""
