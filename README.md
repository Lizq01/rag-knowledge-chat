# 知识库 RAG Agent

把本地资料变成可对话的知识库：**混合检索（BM25 + 语义向量）→ 查询改写 → 重排 → 带引用来源的回答 → 多轮记忆 → Web 界面 → 自动化评测**。

## 核心特性

- **混合检索**：BM25 关键词检索 + BGE 语义向量检索，RRF（倒数排名融合）合并两路结果
- **查询改写**：多轮场景下自动做指代消解（「那这个 k 取多少？」→ 补全主语），并扩展多路查询变体加权融合
- **双后端重排**：本地 Cross-Encoder（bge-reranker，ONNX 运行）或 LLM 打分，可 A/B 切换与自动降级
- **FAISS 向量索引**：`IndexFlatIP` 精确检索 / `IndexHNSWFlat` 近似检索，装不上自动退回 NumPy
- **多轮记忆**：短期滑动窗口（最近 N 轮原文）+ 中期 LLM 摘要压缩 + 会话落盘（重启可恢复、TTL 自动淘汰）
- **Web 界面**：FastAPI + 单文件原生 HTML（无任何 CDN 依赖，断网可用），含并发背压
- **MCP 服务端**：检索层封装为 `search_knowledge` / `get_chunk` / `list_sources` 三个 MCP 工具（stdio，严格 JSON Schema + 凭据隔离），可被任意 MCP Client 复用
- **引用溯源**：回答中的每段事实都标注 `[编号]`，可回查原文出处
- **分层评测**：单轮 30 条 + 多轮 15 条评测集，输出召回率、片段级 MRR、回答要点覆盖率与 P50/P95 延迟
- **轻量本地推理**：fastembed + ONNX Runtime（无需 torch / GPU），模型首次自动下载后离线可用

## 处理链路

```
离线索引：  data/docs/*  →  加载解析  →  语义切分(带重叠)  →  BGE 向量化  →  FAISS 索引 + 元信息
                                                                                │
在线问答：  用户提问  →  查询改写(指代消解+多路扩展) ─┐                             │
                       BM25 召回 ─┐                  │                            │
                       向量召回  ─┼→ RRF 融合(加权) → 重排(CE/LLM) → 带历史生成  │
                                 └──────────────────────────────────────────────┘
多轮记忆：  会话摘要 + 最近 3 轮原文  →  生成上下文；超出阈值自动摘要压缩，落盘 data/sessions/
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

# 6. 提问（单轮）
python main.py ask "RRF 融合算法的公式是什么？"
python main.py ask "RRF 的公式？" --rerank none      # 指定重排后端做对比

# 7. 多轮对话（CLI）：追问可以直接说「那这个 k 一般取多少？」
python main.py chat                                    # /new 开新会话 · /history 看记忆 · /quit 退出

# 8. 启动 Web 界面（默认 http://127.0.0.1:8000）
python main.py web

# 9. 跑评测集 / 性能基准
python main.py eval --compare          # 单轮：对比 none / llm / cross-encoder 重排
python main.py eval --multi-turn       # 多轮：对比查询改写开/关的效果
python main.py bench                   # 向量后端与重排后端的延迟基准
```

## MCP 服务端（把检索能力交给其他 Agent）

MCP（Model Context Protocol）已是智能体工具层的事实标准：Server 用 JSON Schema 声明「有哪些工具、参数什么形状」，
Client 在运行时发现并调用。本项目把检索层原样封成一个 stdio MCP Server，`deep-research-agent` 作为 Client 在 LangGraph 节点里调用它。

```bash
python -m mcp_server            # 启动服务端（通常由 Client 拉起；手动跑会停在等待 stdin）
python -m mcp_server.selfcheck  # 端到端自检：24 项（工具发现 / 正常调用 / 非法参数 / 凭据隔离）

# 官方 inspector（Web UI）：tools 页能看到 3 个工具与完整 Schema，可手工填参数调用
npx @modelcontextprotocol/inspector ./.venv/Scripts/python.exe -m mcp_server
# 无界面时用 CLI 模式，直接打印 tools/list
npx @modelcontextprotocol/inspector --cli ./.venv/Scripts/python.exe -m mcp_server --method tools/list
```

### 三个工具

| 工具 | 必填 | 参数与约束 | 用途 |
|---|---|---|---|
| `search_knowledge` | `query` | `query` 2~200 字；`top_k` 1~20（默认 5）；`mode` ∈ `hybrid`/`bm25`/`vector`；`rerank` ∈ `none`/`cross-encoder` | 混合检索，返回带 `chunk_id` 的片段 |
| `get_chunk` | `chunk_id` | 匹配 `^[^#\s]{1,180}\.(md|txt|pdf)::\d{1,4}$` | 取回完整片段，附前后相邻 id |
| `list_sources` | — | 无参数 | 文档清单、规模与索引信息 |

约束不是注释，是**契约**：`tools/list` 下发的 `inputSchema` 带 `required` / `enum` / `minimum` / `maximum` / `pattern`，
服务端在进业务代码之前就用 pydantic 校验。传 `top_k=999` 不会进入检索函数：

```
Error executing tool search_knowledge: 1 validation error for search_knowledgeArguments
top_k
  Input should be less than or equal to 20 [type=less_than_equal, input_value=999]
```

`MCPServer` 走的是 MCP 的「工具执行错误」通道（`isError=true` 的结果，不是连接层异常），
模型能读到具体哪个字段不合法并自行改正——这正是把 Schema 当契约、而不是当提示的价值。

### 凭据边界

- **服务端自己读 `.env`**：LLM Key、模型路径都在服务端进程；工具签名里没有任何 token / 路径参数；
- **stdio 白名单**：Client 只传进程启动必需的环境变量（SDK 的 `DEFAULT_INHERITED_ENV_VARS` 同理），
  Client 进程里的 `OPENAI_API_KEY` 不会进入服务端进程，`selfcheck` 会断言这一条；
- **出口过滤**：工具只回文档名与片段，不回磁盘路径；异常信息统一过 `scrub()`，把环境变量里的敏感值替换成 `***`；
- **日志分流**：stdout 是 JSON-RPC 通道，日志一律走 stderr（`MCP_LOG_LEVEL` 控制级别）。

### MCP 与传统 Function Calling 的差别

| 维度 | 传统 Function Calling | MCP |
|---|---|---|
| 工具定义从哪来 | 写死在应用代码里，编进 prompt | Server 运行时声明，Client `tools/list` 动态发现 |
| 谁维护 | 每个应用各写一份 | 一个 Server，N 个 Client 复用 |
| 参数契约 | Schema 只是提示，模型传错就是运行时错误 | 同一份 Schema 也是校验规则，服务端先拒 |
| 边界 | 工具在进程内，直接碰应用状态与密钥 | 进程/网络边界，凭据留在 Server 端 |
| 换框架 | 适配层重写 | 协议不变，换 Client 即可 |

一句话：Function Calling 解决「模型怎么表达要调工具」，MCP 解决「工具怎么被发布、发现和复用」。
两者不冲突——本项目的模型仍然用 function calling 决定调哪个工具，只是工具定义来自 MCP Server 而非硬编码。

### stdio 与 HTTP + SSE 怎么选

| | stdio | HTTP + SSE / Streamable HTTP |
|---|---|---|
| 谁启动 | Client 拉起子进程 | Server 独立部署 |
| 鉴权 | 靠进程边界，天然单租户 | 需要 token / OAuth，支持多租户 |
| 部署 | 与 Client 同机，零网络配置 | 可远程、可共享给多个 Client |
| 典型坑 | ① 服务端 `print()` 污染 stdout，JSON-RPC 帧错乱；② 子进程残留（必须显式关会话）；③ 跨任务关闭 anyio cancel scope 会抛 `Attempted to exit cancel scope in a different task`；④ 冷启动（加载索引/模型）压在首次调用上 | ① 长连接被中间层掐断，需心跳与重连；② 无状态代理下 session id 丢失；③ CORS 与鉴权要自己做；④ 多实例时不能把会话状态放进程内存 |

本项目的选择是 **stdio**：同机、单用户、服务端依赖重（FAISS + ONNX 模型），HTTP 化只会多一层鉴权与运维成本。
`KnowledgeService` 与传输无关，要上远程只换外壳（`server.run("streamable-http")`）。

### 自检输出（实测）

```
[1/4] 工具发现（tools/list）
  [PASS] 工具集合 = ['get_chunk', 'list_sources', 'search_knowledge']
  · search_knowledge  必填=['query']
      top_k:integer 范围[1,20]   mode:string enum=['hybrid', 'bm25', 'vector']
[2/4] 正常调用
  [PASS] get_chunk(rag-notes.md::3) 命中
[3/4] 非法参数必须被拒绝
  [PASS] top_k 超上限 → 已拒绝        [PASS] chunk_id 格式不符 → 已拒绝
[4/4] 凭据隔离
  [PASS] Client 传给服务端的 env 不含任何 KEY/TOKEN
结果：24 通过 / 0 失败
```

## 目录结构

```
├── main.py              # CLI 入口（--check / warmup / ingest / ask / chat / web / eval / bench）
├── mcp_server/          # MCP 服务端：把检索能力开放给其他 Agent（stdio）
│   ├── server.py        # 工具注册、JSON Schema、stdio 启动
│   ├── knowledge.py     # KnowledgeService：检索 / 取片段 / 文档清单（与传输无关）
│   ├── config.py        # 服务端配置（全部来自环境变量）
│   └── selfcheck.py     # 端到端自检（工具发现 / 正常调用 / 非法参数 / 凭据隔离）
├── src/
│   ├── config/             # ① 配置层：参数与环境变量的唯一出口
│   │   └── settings.py     # .env 读取、路径常量、HF 镜像环境变量
│   ├── core/               # ② 抽象接口层：只放契约（ABC），不含实现
│   │   ├── llm_client.py   # BaseLLMClient：LLM 客户端接口
│   │   ├── vector_store.py # BaseVectorStore：向量库接口
│   │   └── reranker.py     # BaseReranker：重排器接口
│   ├── modules/            # ③ 具体实现层：实现 core 的契约
│   │   ├── llm.py          # LLM 客户端（显式超时 + 退避重试）
│   │   ├── embedder.py     # 本地向量化（fastembed / BGE）
│   │   ├── ingest.py       # 文档加载与语义切分
│   │   ├── vectorstore.py  # 向量索引（FAISS flat/hnsw + NumPy 兜底）
│   │   ├── store.py        # 索引构建、持久化、元信息与新鲜度校验
│   │   ├── retriever.py    # 混合检索（BM25 + 向量 + RRF 融合，多路加权）
│   │   ├── reranker.py     # 重排（Cross-Encoder / LLM / 不重排，含降级链路）
│   │   ├── rewrite.py      # 查询改写（LLM 指代消解+扩展 / 规则拼接 / 关闭）
│   │   └── memory.py       # 多轮记忆（滑动窗口 + 摘要压缩 + 会话仓库/TTL）
│   ├── services/           # ④ 业务编排层：把检索 / 重排 / 生成串起来
│   │   ├── rag.py          # 问答主流程（单轮 answer / 多轮 answer_turn）
│   │   ├── evaluate.py     # 评测（召回率 + 片段级 MRR + 要点覆盖率 + 多轮对比）
│   │   └── web.py          # FastAPI 应用 + 单文件页面 + 并发背压
│   └── utils/              # ⑤ 通用工具层（占位：重试/日志现内联在 modules）
├── data/docs/           # 知识库文档（放入你的资料）
├── evals/               # 评测集（eval_set 30 条单轮 / multi_turn_set 15 条多轮）
├── index/               # 索引文件（自动生成，不提交 git）
├── data/sessions/       # 会话持久化（自动生成，不提交 git）
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

**为什么要查询改写**：多轮对话里用户的追问充满指代与省略（「那这个 k 取多少？」——BM25 与向量检索都不知道「k」指 RRF 里的平滑常数）。改写分两步：LLM 把上一轮上下文带进来做指代消解，同时生成 2 个语义变体；多路查询各自跑混合检索，结果做**加权 RRF** 融合（原始问题权重 1.0，改写变体 0.7——变体只是扩展视角，不能喧宾夺主）。首轮没有上下文，默认不改写（省一次 LLM 调用）。

**为什么记忆分三层**：最近 3 轮保留原文（多轮正确性高度依赖最近几轮，原文最可靠）；更早的轮次压成不超过 300 字的摘要（避免上下文随对话长度线性膨胀）；整段对话落盘 JSON（进程重启后可恢复）。**压缩触发有两个条件**：字符数超阈值，或轮数逼近硬上限——少了后者，短答案的长对话会永远走不到摘要，直接被静默截断（这是开发中实测踩到的坑）。压缩失败降级为「丢弃较早轮次」而不是报错：少记几轮历史不会让问答失败，抛异常会。

**为什么 Web 要做并发背压**：检索+重排是 CPU 密集活。实测并发 7 个请求时：信号量上限 4 个正常返回（200），3 个快速失败 503「服务繁忙」——同时 4 个重排在 32 核机器上互相抢核，单请求延迟从 5.6s 涨到 13.5s。结论：与其让请求排队把机器拖垮，不如快速失败让前端重试。

## 语料与评测集

`data/docs/` 现有 5 篇笔记：原始 2 篇（`ai-agent-notes.md` / `rag-notes.md`）加 3 篇**干扰文档**（`vector-db-notes.md` / `prompt-engineering-notes.md` / `llm-finetune-notes.md`）。

加干扰文档是必要的：只有 2 篇文档时全部候选只有 9 个片段，任何检索方式都能把正确文档排在第一位，指标全部 100% 饱和，**无法区分重排后端的好坏**。加入主题相近的干扰文档后（17 个片段），片段级 MRR 才真正有了区分度。

评测集 30 条，构成如下：

| 类型 | 条数 | 说明 |
|---|---|---|
| 原文复述型 | 22 | 问题用词与原文高度重合，考察基础召回 |
| 改写/跨文档型 | 8 | 换说法提问（如「两路检索结果要合并成一个排序，有没有不用调分数尺度的办法？」），考察语义检索与抗干扰 |

每条包含 `expected_text`（答案所在片段的一段特征文本）。**判定用片段级而非文档级**：一篇文档有 4~5 个片段，文档级命中几乎总是 100%，无法反映「排在最前面的是不是真正包含答案的那一段」。

多轮评测集 `evals/multi_turn_set.jsonl`（15 条）在单轮格式上多了 `history` 字段（1 轮真实问答上下文），追问全部带指代或省略：

| 追问类型 | 示例 | 条数 |
|---|---|---|
| 代词指代 | 「那这个 k 一般取多少？」（k 指 RRF 平滑常数） | 8 |
| 省略主语 | 「短期记忆呢？它俩有什么区别？」 | 5 |
| 反向提问 | 「反过来，什么情况下该选微调？」 | 2 |

这类问题**不改写时检索必然偏**——「这个 k」「它俩」对 BM25 和向量检索都是无意义的高频词。评测跑法：`python main.py eval --multi-turn`（自动对比改写开/关两组，只测检索与重排，不耗生成 token）。

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

**多轮改写收益（evals/multi_turn_set.jsonl，15 条带上下文的追问）**

| 配置 | 片段 Hit@5 | 片段 MRR | 平均查询路数 | P50 | P95 |
|---|---|---|---|---|---|
| 改写关闭（none） | 93% | 0.856 | 1.0 | 3779 ms | 4934 ms |
| 改写开启（auto，LLM） | **100%** | **0.889** | 3.0 | 4578 ms | 5849 ms |

结论：改写把 7 条指代/省略型追问全部救回（Hit@5 93%→100%），代价是每次多一路 LLM 调用（约 800ms）与 3 倍检索量（检索本身只占几十 ms，可忽略）。**追问占比高的对话场景收益明显；单轮问答完全不受影响**（首轮默认不改写）。

端到端 HTTP 冒烟（Web 界面，同一会话连续三轮）：
- 第 1 轮「RRF 融合算法的公式是什么？」→ 正确给出公式，`改写=none`（首轮不改写）；
- 第 2 轮「那这个 k 一般取多少？」→ 正确答出 k=60，改写查询为 `RRF（倒数排名融合）算法中的平滑常数 k 一般取多少？` 等三路；
- 第 3 轮「它和加权求和比有什么优势？」→ 正确把「它」消解为 RRF，并如实回答知识库中没有直接对比信息（不编造）；
- 并发背压：7 个并发请求 → 4 个 200 + 3 个 503；4 路并发时单请求 P50 从 5.6s 涨到 13.5s（CPU 抢核，重排是瓶颈）。

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
| `REWRITE_BACKEND` | `auto` | `auto` LLM→规则降级 / `llm` / `rule` / `none` |
| `REWRITE_COUNT` | 3 | 每问生成「原始+变体」共几路查询 |
| `REWRITE_ON_FIRST_TURN` | 0 | 首轮（无上下文）是否也改写，默认关 |
| `MEMORY_TURNS` | 3 | 短期窗口：最近 N 轮保留原文 |
| `MEMORY_SUMMARY_CHARS` | 3000 | 历史字符数超此值触发 LLM 摘要压缩 |
| `MEMORY_MAX_TURNS` | 30 | 轮数硬上限（含摘要兜底触发线） |
| `SESSION_MAX_ACTIVE` / `SESSION_TTL_MINUTES` | 50 / 120 | 服务端会话数量上限与空闲过期时间 |
| `MAX_QUESTION_CHARS` | 500 | 单个问题的长度上限（Web 入口校验） |
| `WEB_HOST` / `WEB_PORT` | 127.0.0.1 / 8000 | Web 服务监听地址与端口 |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | 60 / 2 | LLM 调用超时与重试（指数退避 + 抖动） |

## 降级与容错

| 环节 | 失败时 | 表现 |
|---|---|---|
| FAISS 未安装 | 退回 NumPy 暴力检索 | `--check` 标为「可选」，结果不变 |
| FAISS 索引文件损坏 | 报错并提示删除 `index/faiss.index` 重建 | 明确失败，不静默错结果 |
| Cross-Encoder 模型加载失败 | 降级 LLM 重排 | 返回值标记 `degraded_from`，评测统计降级次数 |
| LLM 重排输出不可解析 | 按召回序截断 | 返回 `backend=none` |
| LLM 改写调用失败 | 降级规则改写（拼上一轮问题） | 返回 `degraded_from`，前端显示降级提示 |
| LLM 摘要压缩失败 | 降级丢弃较早轮次 | 返回 `backend=drop`，主流程不中断 |
| 会话文件损坏 | 按新会话处理 | 打 warning，不拒绝服务 |
| 并发超过信号量上限 | 快速返回 503 | 前端提示稍后重试，不排队拖垮机器 |
| 索引与 chunk 数量不一致 | 报错要求重新 ingest | 防止用坏索引回答问题 |
| 文档变更后未重建索引 | 告警提示 | `--check` 输出「建议重新 ingest」 |

## Roadmap

- [x] M1 混合检索 + LLM 重排 + 引用溯源 + 基础评测
- [x] M2 FAISS 向量索引（flat/hnsw 可切换 + NumPy 兜底）、本地 Cross-Encoder 重排、片段级评测与性能基准
- [x] M3 查询改写（指代消解 + 多路扩展 + 加权 RRF）、多轮记忆（窗口 + 摘要 + 会话持久化）、Web 界面（FastAPI + 背压）、多轮评测
- [x] M4 MCP 服务端（search_knowledge / get_chunk / list_sources，严格 Schema + 凭据隔离 + 端到端自检），供 `deep-research-agent` 作为 MCP Client 调用
