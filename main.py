"""知识库 RAG Agent · CLI 入口

用法：
    python main.py --check                     环境自检
    python main.py warmup                      预下载本地模型（向量化 + 重排）
    python main.py ingest                      构建知识库索引（扫描 data/docs/）
    python main.py ask "你的问题"               提问（混合检索 → 重排 → 生成）
    python main.py ask "问题" --rerank none     指定重排后端做 A/B 对比
    python main.py eval                        运行评测集（检索 + 重排 + 回答质量）
    python main.py eval --compare              对比三种重排后端的效果与耗时
    python main.py bench                       性能基准（向量后端 + 重排后端延迟）
"""
import argparse
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

REQUIRED_PACKAGES = [
    ("rich", "终端输出"),
    ("jieba", "中文分词"),
    ("rank_bm25", "BM25 检索"),
    ("numpy", "向量计算"),
    ("fastembed", "本地向量化 + 重排"),
    ("openai", "LLM 调用"),
    ("dotenv", "环境变量"),
    ("pypdf", "PDF 解析"),
]
OPTIONAL_PACKAGES = [
    ("faiss", "向量索引（FAISS，缺失时自动退回 NumPy）"),
]

RERANK_CHOICES = ["auto", "cross-encoder", "llm", "none"]


def _rerank_model_info() -> tuple[bool, str]:
    """检查配置的重排模型是否被 fastembed 支持（不触发下载）"""
    from src import config
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except Exception as e:  # noqa: BLE001
        return False, f"fastembed 未提供 Cross-Encoder：{type(e).__name__}: {e}"
    for model in TextCrossEncoder.list_supported_models():
        if model["model"].lower() == config.RERANK_MODEL.lower():
            return True, f"{model['model']}（{model['size_in_GB']} GB，首次使用自动下载）"
    return False, f"{config.RERANK_MODEL} 不在 fastembed 支持的重排模型列表中"


def cmd_check() -> int:
    """环境自检"""
    console.print("\n[bold]1) 依赖检查[/bold]")
    ok_all = True
    for mod, desc in REQUIRED_PACKAGES:
        try:
            __import__(mod)
            console.print(f"   [green]OK[/green]    {mod:<12} {desc}")
        except ImportError:
            ok_all = False
            console.print(f"   [red]缺失[/red]  {mod:<12} {desc}")
    for mod, desc in OPTIONAL_PACKAGES:
        try:
            __import__(mod)
            console.print(f"   [green]OK[/green]    {mod:<12} {desc}")
        except ImportError:
            console.print(f"   [yellow]可选[/yellow]  {mod:<12} {desc}")

    console.print("\n[bold]2) 配置检查（.env）[/bold]")
    from src import config
    if config.OPENAI_API_KEY:
        console.print("   [green]OK[/green]    OPENAI_API_KEY 已配置")
    else:
        ok_all = False
        console.print("   [red]缺失[/red]  OPENAI_API_KEY —— 请把 .env.example 复制为 .env 并填写")
    console.print(f"   [green]OK[/green]    MODEL_NAME = {config.MODEL_NAME}")
    console.print(f"   [green]OK[/green]    OPENAI_BASE_URL = {config.OPENAI_BASE_URL}")
    console.print(f"   [green]OK[/green]    EMBEDDING_MODEL = {config.EMBEDDING_MODEL}")
    console.print(f"   [green]OK[/green]    VECTOR_BACKEND = {config.VECTOR_BACKEND}"
                  f"（索引类型 {config.FAISS_INDEX_TYPE}）")

    console.print("\n[bold]3) 重排后端[/bold]")
    console.print(f"   [green]OK[/green]    RERANK_BACKEND = {config.RERANK_BACKEND}")
    supported, note = _rerank_model_info()
    if supported:
        console.print(f"   [green]OK[/green]    RERANK_MODEL = {note}")
    else:
        console.print(f"   [yellow]提示[/yellow]  RERANK_MODEL = {note}")
        console.print("            可设置 RERANK_BACKEND=llm 改用大模型重排")

    console.print("\n[bold]4) 数据与索引[/bold]")
    from src import ingest, store
    docs = ingest.load_documents() if config.DATA_DIR.exists() else []
    console.print(f"   文档数量：{len(docs)} 篇"
                  + ("（把 .md / .txt / .pdf 放进 data/docs/）" if not docs else ""))
    meta = store.index_meta()
    if meta:
        console.print(f"   索引状态：[green]已构建[/green]（{meta['ntotal']} 个片段，"
                      f"维度 {meta['dim']}，向量后端 {meta['backend']}）")
        console.print(f"   构建时间：{meta['created_at']}")
        for warning in store.check_freshness():
            console.print(f"   [yellow]注意[/yellow]  {warning}")
    else:
        console.print("   索引状态：[yellow]未构建[/yellow]（运行 python main.py ingest）")

    console.print()
    if ok_all:
        console.print("[green bold]环境检查通过，可以开始使用了。[/green bold]\n")
        return 0
    console.print("[red bold]环境不完整，请按上面提示修复后重试。[/red bold]\n")
    return 1


def cmd_warmup() -> int:
    """预下载本地模型，避免首次提问时长时间等待"""
    from src import embedder
    from src.reranker import preload_cross_encoder

    console.print("\n[bold]1) 向量化模型[/bold]")
    started = time.perf_counter()
    vector = embedder.embed_query("预热")
    console.print(f"   [green]OK[/green]    {embedder.model_name()} "
                  f"（维度 {vector.shape[0]}，{time.perf_counter() - started:.1f}s）")

    console.print("\n[bold]2) 重排模型[/bold]")
    with console.status("正在下载/加载 Cross-Encoder 重排模型（首次约 0.6GB，请耐心等待）..."):
        ok, note = preload_cross_encoder()
    if ok:
        console.print(f"   [green]OK[/green]    {note}")
    else:
        console.print(f"   [yellow]跳过[/yellow]  {note}")
    console.print()
    return 0 if ok else 1


def cmd_ingest(backend: str | None, index_type: str | None) -> int:
    """构建索引"""
    from src import store
    with console.status("正在扫描文档并构建索引（首次运行会下载向量模型）..."):
        stats = store.build_index(backend=backend, index_type=index_type)
    console.print(f"\n[green bold]索引构建完成[/green bold]"
                  f"：{stats['docs']} 篇文档 → {stats['chunks']} 个片段，"
                  f"向量维度 {stats['dim']}")
    console.print(f"向量后端：[bold]{stats['backend']}[/bold]"
                  f"（写入文件：{', '.join(stats['saved'])}）\n")
    return 0


def cmd_ask(question: str, rerank_backend: str) -> int:
    """提问"""
    from src import rag
    with console.status("检索并生成回答中..."):
        result = rag.answer(question, rerank_backend=rerank_backend)

    console.print()
    console.print(Panel(Markdown(result["answer"]),
                        title="回答", border_style="green", padding=(1, 2)))

    if result["citations"]:
        table = Table(title="引用来源", show_lines=True, expand=True)
        table.add_column("#", width=3, justify="center")
        table.add_column("文档", style="cyan", no_wrap=True)
        table.add_column("位置", width=8, justify="center")
        table.add_column("原文片段", overflow="fold")
        for c in result["citations"]:
            snippet = c["text"][:120].replace("\n", " ")
            table.add_row(str(c["id"]), c["doc"], f"第 {c['position']} 段",
                          snippet + ("..." if len(c["text"]) > 120 else ""))
        console.print(table)

    info = result["rerank"]
    timing = result["timings"]
    load_note = (f"（含模型加载 {info['model_load_ms']}ms）"
                 if info.get("model_load_ms") else "")
    line = (f"向量后端 [bold]{result['vector_backend']}[/bold]"
            + f"｜重排后端 [bold]{info['backend']}[/bold]"
            + (f"（降级自 {info['degraded_from']}）" if info.get("degraded_from") else "")
            + f"｜检索 {timing['retrieve_ms']:.0f}ms"
            + f"｜重排 {timing['rerank_ms']:.0f}ms{load_note}"
            + f"｜生成 {timing['llm_ms']:.0f}ms"
            + f"｜总计 {timing['total_ms']:.0f}ms")
    console.print(line)
    if info.get("reason"):
        console.print(f"[yellow]降级原因：{info['reason']}[/yellow]")
    console.print()
    return 0


def cmd_bench(repeat: int, scale: int, rerank_backends: list[str]) -> int:
    """性能基准：向量后端检索延迟 + 重排后端延迟

    「升级索引/换重排模型」这类改动必须量化，否则只是心理安慰。
    本命令输出 P50/P95 延迟，并给出 HNSW 相对精确检索的召回率。
    """
    import numpy as np

    from src import embedder, evaluate, store
    from src.config import INDEX_DIR, TOP_K
    from src.retriever import HybridRetriever
    from src.vectorstore import FaissIndex, NumpyIndex, faiss_available

    retriever = HybridRetriever()
    chunks = retriever.chunks
    console.print(f"\n语料：{len(chunks)} 个片段｜当前索引后端：{retriever.backend}\n")

    # ---- A. 向量后端检索延迟（真实语料）----
    vectors = np.load(INDEX_DIR / "vectors.npy")
    queries = [item["question"] for item in evaluate.load_eval_set()]
    query_vecs = embedder.embed_texts(queries)

    backends: dict[str, object] = {"numpy": NumpyIndex(vectors)}
    ok, _ = faiss_available()
    if ok:
        backends["faiss:flat"] = FaissIndex.build(vectors, "flat")
        backends["faiss:hnsw"] = FaissIndex.build(vectors, "hnsw")

    table = Table(title=f"A) 向量检索延迟（真实语料 {len(chunks)} 条，查询 {len(queries)} 条 × {repeat} 轮）",
                  show_lines=True)
    table.add_column("后端", width=14)
    table.add_column("P50 (ms)", width=10, justify="center")
    table.add_column("P95 (ms)", width=10, justify="center")
    table.add_column("相对精确检索召回率", width=18, justify="center")

    exact_top: list[list[int]] = []
    for name, index in backends.items():
        latencies: list[float] = []
        tops: list[list[int]] = []
        for _ in range(repeat):
            for vec in query_vecs:
                started = time.perf_counter()
                hits = index.search(vec, min(TOP_K, len(chunks)))  # type: ignore[attr-defined]
                latencies.append((time.perf_counter() - started) * 1000)
                if len(tops) < len(queries):
                    tops.append([i for i, _ in hits])
        if name == "numpy":
            exact_top = tops
        overlap = 0.0
        if exact_top:
            total = sum(len(set(a) & set(b)) for a, b in zip(exact_top, tops))
            overlap = total / max(1, sum(len(a) for a in exact_top))
        table.add_row(name if name != "numpy" else "numpy（基线）",
                      f"{evaluate.percentile(latencies, 50):.3f}",
                      f"{evaluate.percentile(latencies, 95):.3f}",
                      "100%" if name == "numpy" else f"{overlap:.0%}")
    console.print(table)

    # ---- B. 规模扩展测试（合成向量）----
    if ok and scale > len(chunks):
        # 用「簇中心 + 噪声」合成，而不是纯均匀随机：真实 embedding 有语义簇结构，
        # 纯随机向量在高维空间中距离几乎无差异，会让近似索引的召回率失真（实测仅 36%）。
        rng = np.random.default_rng(42)
        dim = vectors.shape[1]
        centers = rng.random((max(8, scale // 200), dim), dtype=np.float32) - 0.5
        owner = rng.integers(0, centers.shape[0], scale)
        big = centers[owner] + (rng.random((scale, dim), dtype=np.float32) - 0.5) * 0.6
        big = np.ascontiguousarray(big, dtype=np.float32)
        big /= np.linalg.norm(big, axis=1, keepdims=True)
        probes = big[rng.choice(scale, 20, replace=False)] + (
            rng.random((20, dim), dtype=np.float32) - 0.5) * 0.3
        probes = np.ascontiguousarray(probes, dtype=np.float32)
        probes /= np.linalg.norm(probes, axis=1, keepdims=True)

        table2 = Table(title=f"B) 规模扩展测试（合成 {scale} 条 {vectors.shape[1]} 维向量，20 次查询取均值）",
                       show_lines=True)
        table2.add_column("后端", width=14)
        table2.add_column("单次查询 (ms)", width=14, justify="center")
        table2.add_column("相对精确检索召回率", width=18, justify="center")
        flat_index = FaissIndex.build(big, "flat")
        hnsw_index = FaissIndex.build(big, "hnsw")
        # 逐条查询分别取精确基线，避免用同一条基线去比所有查询
        baselines = [[i for i, _ in flat_index.search(p, TOP_K)] for p in probes]
        for name, index in (("numpy", NumpyIndex(big)), ("faiss:flat", flat_index),
                            ("faiss:hnsw", hnsw_index)):
            started = time.perf_counter()
            hits_all = [index.search(p, TOP_K) for p in probes]
            per_query = (time.perf_counter() - started) * 1000 / len(probes)
            overlap = sum(len(set(b) & {i for i, _ in h})
                          for b, h in zip(baselines, hits_all)) / (len(probes) * TOP_K)
            table2.add_row(name, f"{per_query:.2f}", f"{overlap:.0%}")
        console.print(table2)
        console.print("[dim]注：合成数据仅用于观察延迟量级；近似索引的真实召回率必须在真实 "
                      "embedding 上测量，并按需调大 FAISS_HNSW_EF_SEARCH[/dim]")

    # ---- C. 重排后端延迟 ----
    if rerank_backends:
        table3 = Table(title=f"C) 重排延迟（每条问题重排 {TOP_K} 个候选）", show_lines=True)
        table3.add_column("重排后端", width=16)
        table3.add_column("P50 (ms)", width=10, justify="center")
        table3.add_column("P95 (ms)", width=10, justify="center")
        for backend in rerank_backends:
            rows = evaluate.eval_rerank(retriever, evaluate.load_eval_set(), backend)
            table3.add_row(rows["used_backend"],
                           f"{rows['latency_ms_p50']:.0f}", f"{rows['latency_ms_p95']:.0f}")
        console.print(table3)
    console.print()
    return 0


def _print_answer_table(answers: list[dict]) -> float:
    table = Table(title="3) 回答要点覆盖率（LLM 裁判）", show_lines=True)
    table.add_column("问题", overflow="fold", max_width=44)
    table.add_column("覆盖要点", overflow="fold", max_width=30)
    table.add_column("覆盖分", width=8, justify="center")
    for row in answers:
        j = row["judge"]
        color = "green" if j["score"] >= 0.8 else ("yellow" if j["score"] >= 0.5 else "red")
        table.add_row(row["question"],
                      "；".join(j["covered"]) or "-",
                      f"[{color}]{j['score']:.0%}[/{color}]")
    console.print(table)
    return sum(r["judge"]["score"] for r in answers) / len(answers)


def cmd_eval(rerank_backend: str, compare: bool, skip_answers: bool) -> int:
    """运行评测集"""
    from src import evaluate
    from src.config import RERANK_TOP_N, TOP_K
    from src.retriever import HybridRetriever

    items = evaluate.load_eval_set()
    console.print(f"\n评测集：{len(items)} 条问题\n")

    retriever = HybridRetriever()
    console.print(f"向量后端：[bold]{retriever.backend}[/bold]\n")

    # ---- 第一层：检索质量 ----
    with console.status("正在评测检索质量（Recall@K）..."):
        retrieval = evaluate.eval_retrieval(retriever, items)

    table = Table(title=f"1) 召回能力（答案所在片段是否进前 {TOP_K} 条）", show_lines=True)
    table.add_column("问题", overflow="fold", max_width=40)
    table.add_column("期望文档", style="cyan", no_wrap=True)
    table.add_column("命中", width=6, justify="center")
    table.add_column("片段排名", width=8, justify="center")
    for row in retrieval:
        table.add_row(row["question"], row["expected_doc"],
                      "[green]是[/green]" if row["hit"] else "[red]否[/red]",
                      str(row["rank"] or "-"))
    console.print(table)
    recall = sum(r["hit"] for r in retrieval) / len(retrieval)
    console.print(f"\n片段级 Recall@{TOP_K} = [bold]{recall:.0%}[/bold]"
                  f"（{sum(r['hit'] for r in retrieval)}/{len(retrieval)}）\n")

    # ---- 第二层：重排质量 ----
    backends = ["none", "llm", "cross-encoder"] if compare else [rerank_backend]
    with console.status("正在评测重排质量（Hit@N / MRR）..."):
        rerank_rows = evaluate.compare_backends(retriever, items, backends)

    table2 = Table(title=f"2) 重排质量（重排后 Top-{RERANK_TOP_N}）", show_lines=True)
    table2.add_column("重排后端", width=18)
    table2.add_column(f"片段 Hit@{RERANK_TOP_N}", width=12, justify="center")
    table2.add_column("片段 MRR", width=9, justify="center")
    table2.add_column(f"文档 Hit@{RERANK_TOP_N}", width=12, justify="center")
    table2.add_column("P50 (ms)", width=9, justify="center")
    table2.add_column("P95 (ms)", width=9, justify="center")
    table2.add_column("降级", width=6, justify="center")
    for row in rerank_rows:
        degraded = row["degraded"]
        label = row["used_backend"]
        if degraded:
            label = f"{label} [yellow](降级{degraded})[/yellow]"
        table2.add_row(label, f"{row['hit_at_n']:.0%}", f"{row['mrr']:.3f}",
                       f"{row['doc_hit_at_n']:.0%}",
                       f"{row['latency_ms_p50']:.0f}", f"{row['latency_ms_p95']:.0f}",
                       str(degraded))
    console.print(table2)
    console.print("[dim]片段级指标是主判据；文档级通常已饱和，仅作参考[/dim]\n")

    # ---- 第三层：回答质量 ----
    if skip_answers:
        console.print("[dim]已跳过回答质量评测（--skip-answers）[/dim]\n")
        return 0

    with console.status("正在评测回答质量（LLM 裁判要点覆盖）..."):
        answers = evaluate.eval_answers(items, retriever, rerank_backend=rerank_backend)

    avg = _print_answer_table(answers)
    console.print(f"\n平均要点覆盖率 = [bold]{avg:.0%}[/bold]\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="知识库 RAG Agent（混合检索 + 重排 + 引用溯源 + 评测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--check", action="store_true", help="环境自检")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("warmup", help="预下载本地模型（向量化 + Cross-Encoder 重排）")

    ingest_parser = sub.add_parser("ingest", help="构建知识库索引")
    ingest_parser.add_argument("--backend", choices=["auto", "faiss", "numpy"],
                               help="向量索引后端（默认取 .env 的 VECTOR_BACKEND）")
    ingest_parser.add_argument("--index-type", choices=["flat", "hnsw"],
                               dest="index_type",
                               help="FAISS 索引类型：flat 精确 / hnsw 近似")

    ask_parser = sub.add_parser("ask", help="提问")
    ask_parser.add_argument("question", help="要问的问题")
    ask_parser.add_argument("--rerank", choices=RERANK_CHOICES, default=None,
                            help="重排后端（默认取 .env 的 RERANK_BACKEND）")
    ask_parser.add_argument("--no-rerank", action="store_true",
                            help="关闭重排（等价于 --rerank none）")

    eval_parser = sub.add_parser("eval", help="运行评测集")
    eval_parser.add_argument("--rerank", choices=RERANK_CHOICES, default=None,
                             help="重排后端（默认取 .env 的 RERANK_BACKEND）")
    eval_parser.add_argument("--compare", action="store_true",
                             help="对比 none / llm / cross-encoder 三种重排后端")
    eval_parser.add_argument("--skip-answers", action="store_true",
                             help="只评测检索与重排，不调用 LLM 裁判（省 token）")

    bench_parser = sub.add_parser("bench", help="性能基准（向量后端 + 重排后端延迟）")
    bench_parser.add_argument("--repeat", type=int, default=3, help="检索延迟测试轮数")
    bench_parser.add_argument("--scale", type=int, default=20000,
                              help="规模扩展测试的合成向量条数（0 表示跳过）")
    bench_parser.add_argument("--rerank", nargs="*", default=None,
                              choices=RERANK_CHOICES,
                              help="要测试的重排后端，默认全部（含 llm 会产生 API 调用）")

    args = parser.parse_args()

    if args.check:
        return cmd_check()
    if args.command == "bench":
        backends = args.rerank or ["none", "cross-encoder"]
        return cmd_bench(args.repeat, args.scale, backends)
    if args.command == "warmup":
        return cmd_warmup()
    if args.command == "ingest":
        return cmd_ingest(args.backend, args.index_type)
    if args.command == "ask":
        backend = "none" if args.no_rerank else args.rerank
        return cmd_ask(args.question, backend)
    if args.command == "eval":
        return cmd_eval(args.rerank or "auto", args.compare, args.skip_answers)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
