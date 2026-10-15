# 知识库 RAG Agent

把本地资料变成可对话的知识库：**混合检索（BM25 + 语义向量）→ LLM 重排 → 带引用来源的回答 → 自动化评测**。

## 核心特性

- **混合检索**：BM25 关键词检索 + BGE 语义向量检索，RRF（倒数排名融合）合并两路结果
- **LLM 重排**：对召回候选做相关性精排，过滤噪声、提升引用精度
- **引用溯源**：回答中的每段事实都标注 `[编号]`，可回查原文出处
- **量化评测**：内置评测集，输出检索命中率（Recall@K）与回答要点覆盖率（LLM 裁判）
- **轻量本地向量化**：fastembed + ONNX 运行时（无需 torch / GPU），中文 BGE 模型首次自动下载后离线可用

## 处理链路

```
离线索引：  data/docs/*  →  加载解析  →  语义切分(带重叠)  →  BGE 向量化  →  索引持久化
                                                                                │
在线问答：  用户提问  →  BM25 召回 ─┐                                            │
                       向量召回  ─┼→ RRF 融合 → LLM 重排 → 生成带引用的回答      │
                                 └──────────────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 安装依赖（已预先装好）
pip install -r requirements.txt

# 2. 配置 .env（已预生成，只需确认 API Key）
#    把 .env.example 复制为 .env 并填写 OPENAI_API_KEY

# 3. 环境自检
python main.py --check

# 4. 放入资料（支持 .md / .txt / .pdf），构建索引
python main.py ingest

# 5. 提问
python main.py ask "RRF 融合算法的公式是什么？"

# 6. 跑评测集
python main.py eval
```

## 目录结构

```
├── main.py              # CLI 入口（--check / ingest / ask / eval）
├── src/
│   ├── config.py        # 配置与路径
│   ├── llm.py           # LLM 客户端（OpenAI 兼容接口）
│   ├── ingest.py        # 文档加载与语义切分
│   ├── embedder.py      # 本地向量化（fastembed / BGE）
│   ├── store.py         # 索引构建与持久化
│   ├── retriever.py     # 混合检索（BM25 + 向量 + RRF 融合）
│   ├── reranker.py      # LLM 重排
│   ├── rag.py           # 问答主流程（生成带引用的回答）
│   └── evaluate.py      # 评测（Recall@K + 要点覆盖率）
├── data/docs/           # 知识库文档（放入你的资料）
├── evals/eval_set.jsonl # 评测集
├── index/               # 索引文件（自动生成，不提交 git）
└── outputs/             # 运行产物
```

## 关键设计说明

**为什么混合检索**：BM25 擅长精确匹配（术语、编号），向量检索擅长语义改写，两者互补。RRF 只用排名不用分数，天然规避两路评分尺度不一致的问题。

**为什么先召回后重排**：召回阶段追求「不漏」（返回 10~20 条候选），重排阶段追求「精准」（Cross-Encoder / LLM 精排出 top 5），兼顾速度与精度。

**切分策略**：先按段落聚合、超长段落按句子边界二次切分、相邻块保留重叠区，避免语义在边界被切断。

**评测驱动**：`evals/eval_set.jsonl` 保存固定的「问题 - 期望文档 - 要点」三元组，任何参数调整（切分长度、top_k、开/关重排）都可通过 `python main.py eval` 量化验证。

## Roadmap

- [x] M1 混合检索 + LLM 重排 + 引用溯源 + 基础评测
- [ ] M2 升级向量索引（FAISS / Chroma）、接入 Cross-Encoder 重排（bge-reranker）
- [ ] M3 查询改写（Query Rewriting）、多轮对话记忆、Web 界面
