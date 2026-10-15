# 知识库 RAG Agent

把本地资料变成可对话的知识库：**混合检索（BM25 + 语义向量）→ 重排 → 带引用来源的回答 → 自动化评测**。

## 核心特性

- **混合检索**：BM25 关键词检索 + BGE 语义向量检索，RRF（倒数排名融合）合并两路结果
- **双后端重排**：本地 Cross-Encoder（bge-reranker，ONNX 运行）或 LLM 打分，可 A/B 切换与自动降级
- **FAISS 向量索引**：`IndexFlatIP` 精确检索 / `IndexHNSWFlat` 近似检索，装不上自动退回 NumPy
- **引用溯源**：回答中的每段事实都标注 `[编号]`，可回查原文出处
- **分层评测**：30 条评测集，输出召回率、片段级 MRR、回答要点覆盖率与 P50/P95 延迟
- **轻量本地推理**：fastembed + ONNX Runtime（无需 torch / GPU），模型首次自动下载后离线可用

## 处理链路

```
离线索引：  data/docs/*  →  加载解析  →  语义切分(带重叠)  →  BGE 向量化  →  FAISS 索引 + 元信息
                                                                                │
在线问答：  用户提问  →  BM25 召回 ─┐                                            │
                       向量召回  ─┼→ RRF 融合 → 重排(CE/LLM) → 生成带引用的回答  │
                                 └──────────────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 .env（把 .env.example 复制为 .env 并填写 OPENAI_API_KEY）

# 3. 环境自检
python main.py --check

# 4. 预下载本地模型（向量化 + Cross-Encoder 重排，首次约 0.7GB，可选但推荐）
python main.py warmup

# 5. 放入资料（支持 .md / .txt / .pdf），构建索引
python main.py ingest

# 6. 提问
python main.py ask "RRF 融合算法的公式是什么？"
python main.py ask "RRF 的公式？" --rerank none      # 指定重排后端做对比

# 7. 跑评测集 / 性能基准
python main.py eval --compare          # 对比 none / llm / cross-encoder
python main.py bench                   # 向量后端与重排后端的延迟基准
```

## 目录结构

```
├── main.py              # CLI 入口（--check / warmup / ingest / ask / eval / bench）
├── src/
│   ├── config.py        # 配置与路径
│   ├── llm.py           # LLM 客户端（显式超时 + 退避重试）
│   ├── ingest.py        # 文档加载与语义切分
│   ├── embedder.py      # 本地向量化（fastembed / BGE）
│   ├── vectorstore.py   # 向量索引抽象（FAISS flat/hnsw + NumPy 兜底）
│   ├── store.py         # 索引构建、持久化、元信息与新鲜度校验
│   ├── retriever.py     # 混合检索（BM25 + 向量 + RRF 融合）
│   ├── reranker.py      # 重排（Cross-Encoder / LLM / 不重排，含降级链路）
│   ├── rag.py           # 问答主流程（生成带引用的回答）
│   └── evaluate.py      # 评测（召回率 + 片段级 MRR + 要点覆盖率）
├── data/docs/           # 知识库文档（放入你的资料）
├── evals/eval_set.jsonl # 评测集（30 条，含片段级 ground truth）
├── index/               # 索引文件（自动生成，不提交 git）
└── outputs/             # 运行产物
```

## 关键设计说明

**为什么混合检索**：BM25 擅长精确匹配（术语、编号），向量检索擅长语义改写，两者互补。RRF 只用排名不用分数，天然规避两路评分尺度不一致的问题。

**为什么先召回后重排**：召回阶段追求「不漏」（返回 10~20 条候选），重排阶段追求「精准」（精排出 top 5），兼顾速度与精度。

**为什么默认用本地 Cross-Encoder 而不是 LLM 重排**：实测（30 条评测集）Cross-Encoder 的片段级 MRR 为 **0.983**，不重排 0.917，LLM 重排 **0.873**——LLM 重排在本评测集上反而低于不重排（它会把「看起来相关」的噪声片段提前）。加上 Cross-Encoder 不消耗 token、无网络依赖、结果可复现，因此作为默认；LLM 重排保留为降级路径与对照后端。

**为什么向量索引要抽象一层**：FAISS 是可选依赖。平台没有对应 wheel 时必须能退回 NumPy 而不中断服务，同时 `flat`（精确）与 `hnsw`（近似）要在同一契约下可切换，因此抽出 `VectorIndex`（`backend` / `dim` / `ntotal` / `search`），后端选择集中在 `vectorstore.build_vector_index()`。

**为什么索引要记元信息**：RAG 最隐蔽的线上事故是「文档改了但索引没重建」——回答看起来正常，其实检索的是旧内容。`index/meta.json` 记录文档指纹、向量模型与切分参数，加载时自动比对并告警。

**切分策略**：先按段落聚合、超长段落按句子边界二次切分、相邻块保留重叠区，避免语义在边界被切断。

**评测驱动**：`evals/eval_set.jsonl` 保存「问题 - 期望文档 - 答案片段特征文本 - 要点」四元组。任何参数调整（切分长度、top_k、重排后端）都可通过 `python main.py eval --compare` 量化验证。

## 语料与评测集

`data/docs/` 现有 5 篇笔记：原始 2 篇（`ai-agent-notes.md` / `rag-notes.md`）加 3 篇**干扰文档**（`vector-db-notes.md` / `prompt-engineering-notes.md` / `llm-finetune-notes.md`）。

加干扰文档是必要的：只有 2 篇文档时全部候选只有 9 个片段，任何检索方式都能把正确文档排在第一位，指标全部 100% 饱和，**无法区分重排后端的好坏**。加入主题相近的干扰文档后（17 个片段），片段级 MRR 才真正有了区分度。

评测集 30 条，构成如下：

| 类型 | 条数 | 说明 |
|---|---|---|
| 原文复述型 | 22 | 问题用词与原文高度重合，考察基础召回 |
| 改写/跨文档型 | 8 | 换说法提问（如「两路检索结果要合并成一个排序，有没有不用调分数尺度的办法？」），考察语义检索与抗干扰 |

每条包含 `expected_text`（答案所在片段的一段特征文本）。**判定用片段级而非文档级**：一篇文档有 4~5 个片段，文档级命中几乎总是 100%，无法反映「排在最前面的是不是真正包含答案的那一段」。

## 实测基线（2026-10-15，Windows 11 / Python 3.13 / 32 核 CPU / 无 GPU）

语料：5 篇中文技术笔记 → 17 个片段；评测集 30 条问题。

**重排质量（重排后 Top-5）**

| 重排后端 | 片段 Hit@5 | 片段 MRR | 文档 Hit@5 | P50 | P95 |
|---|---|---|---|---|---|
| none（不重排） | 100% | 0.917 | 100% | 0 ms | 0 ms |
| llm（DeepSeek 打分） | 100% | 0.890 | 100% | 960 ms | 1190 ms |
| cross-encoder（bge-reranker-v2-m3-int8） | 100% | **0.983** | 100% | 3896 ms | 4514 ms |

（片段级 Recall@10 = 100%，30/30；端到端平均要点覆盖率 99%。LLM 重排的分数每次跑会有 ±0.02 波动，Cross-Encoder 完全可复现。）

**向量检索延迟（真实语料 17 条）**

| 后端 | P50 | P95 | 说明 |
|---|---|---|---|
| numpy（暴力，基线） | 0.029 ms | 0.032 ms | 基线 |
| faiss:flat | 0.021 ms | 0.023 ms | 结果与 numpy 完全一致（同为精确检索） |
| faiss:hnsw | 0.027 ms | 0.030 ms | 近似检索，本规模下无优势 |

**规模扩展测试（2 万条 512 维带簇结构合成向量，20 次查询均值）**

| 后端 | 单次查询 | 相对精确检索召回率 |
|---|---|---|
| numpy（多线程 BLAS） | 3.20 ms | 100%（基线） |
| faiss:flat（单线程） | 12.72 ms | 100%（与基线完全一致） |
| faiss:hnsw（efSearch=64） | **0.51 ms** | 72%（合成数据，见下） |

结论要说清楚：

1. **当前语料规模下（17 个片段）FAISS 的收益可以忽略**，numpy 的 `argpartition` 已经足够快。FAISS 的价值在于规模扩展与工程一致性（同一套接口换后端不动业务代码），不是当下的性能。
2. **`faiss:flat` 在这里比 numpy 慢**，因为 faiss-cpu 默认单线程，而 numpy 走的是多线程 BLAS。规模再往上（百万级、内存放不下）结论才会反转。
3. **HNSW 靠算法复杂度取胜**（2 万条下比暴力检索快一个量级），代价是近似：合成数据上召回率只有 72%。**近似索引的召回率必须在真实 embedding 上测**，并通过 `FAISS_HNSW_EF_SEARCH` 调到可接受为止——这也是为什么默认用 `flat`（精确检索，结果与升级前完全一致），`hnsw` 是显式开关而不是默认行为。

**重排模型的选型实测**：`BAAI/bge-reranker-v2-m3-int8`（0.57GB）与 `BAAI/bge-reranker-base`（1.04GB）在本评测集上片段级 MRR 均为 0.983，延迟互有胜负（10 候选重排 3.5s vs 3.7s；换一轮测量又反过来，属同一量级）。因此按模型体积与多语言覆盖选前者。

**需要接受的代价**：Cross-Encoder 每次问答固定增加约 3.5 秒 CPU 计算。交互式场景若嫌慢，可选：
- `RERANK_BACKEND=llm`（约 1 秒，但本评测集上排序质量更差）；
- 调小 `RERANK_MAX_CHARS`（如 300）可把重排耗时降到约 2 秒，代价是长片段尾部不参与打分；
- 离线批处理场景保持默认（质量优先）。

## 配置说明（.env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `VECTOR_BACKEND` | `auto` | `auto` 优先 FAISS / `faiss` 强制 / `numpy` 强制 |
| `FAISS_INDEX_TYPE` | `flat` | `flat` 精确检索；`hnsw` 近似检索（万级以上再考虑） |
| `FAISS_HNSW_M` / `FAISS_HNSW_EF_SEARCH` | 32 / 64 | HNSW 构图与查询期参数，越大越准越慢 |
| `RERANK_BACKEND` | `auto` | `auto` CE→LLM 降级 / `cross-encoder` / `llm` / `none` |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3-int8` | fastembed 内置的 ONNX 重排模型 |
| `RERANK_MAX_CHARS` | 1500 | 单片段送入重排模型的最大字符数 |
| `RERANK_THREADS` | 0（自动） | ONNX 线程数；32 核机器上显式 16 线程比默认快约 23% |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | 60 / 2 | LLM 调用超时与重试（指数退避 + 抖动） |

## 降级与容错

| 环节 | 失败时 | 表现 |
|---|---|---|
| FAISS 未安装 | 退回 NumPy 暴力检索 | `--check` 标为「可选」，结果不变 |
| FAISS 索引文件损坏 | 报错并提示删除 `index/faiss.index` 重建 | 明确失败，不静默错结果 |
| Cross-Encoder 模型加载失败 | 降级 LLM 重排 | 返回值标记 `degraded_from`，评测统计降级次数 |
| LLM 重排输出不可解析 | 按召回序截断 | 返回 `backend=none` |
| 索引与 chunk 数量不一致 | 报错要求重新 ingest | 防止用坏索引回答问题 |
| 文档变更后未重建索引 | 告警提示 | `--check` 输出「建议重新 ingest」 |

## Roadmap

- [x] M1 混合检索 + LLM 重排 + 引用溯源 + 基础评测
- [x] M2 FAISS 向量索引（flat/hnsw 可切换 + NumPy 兜底）、本地 Cross-Encoder 重排、片段级评测与性能基准
- [ ] M3 查询改写（Query Rewriting）、多轮对话记忆、Web 界面
