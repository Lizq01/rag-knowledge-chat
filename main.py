"""知识库 RAG Agent · CLI 入口

用法：
    python main.py --check               环境自检
    python main.py ingest                构建知识库索引（扫描 data/docs/）
    python main.py ask "你的问题"         提问（混合检索 → 重排 → 生成）
    python main.py ask "问题" --no-rerank 提问（关闭重排，用于对比效果）
    python main.py eval                  运行评测集（检索命中率 + 要点覆盖率）
"""
import argparse
import sys

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
    ("fastembed", "本地向量化"),
    ("openai", "LLM 调用"),
    ("dotenv", "环境变量"),
    ("pypdf", "PDF 解析"),
]


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

    console.print("\n[bold]3) 数据与索引[/bold]")
    from src import ingest, store
    docs = ingest.load_documents() if config.DATA_DIR.exists() else []
    console.print(f"   文档数量：{len(docs)} 篇"
                  + ("（把 .md / .txt / .pdf 放进 data/docs/）" if not docs else ""))
    if (config.INDEX_DIR / "chunks.json").exists():
        chunks, vectors, _ = store.load_index()
        console.print(f"   索引状态：[green]已构建[/green]（{len(chunks)} 个片段，维度 {vectors.shape[1]}）")
    else:
        console.print("   索引状态：[yellow]未构建[/yellow]（运行 python main.py ingest）")

    console.print()
    if ok_all:
        console.print("[green bold]环境检查通过，可以开始使用了。[/green bold]\n")
        return 0
    console.print("[red bold]环境不完整，请按上面提示修复后重试。[/red bold]\n")
    return 1


def cmd_ingest() -> int:
    """构建索引"""
    from src import store
    with console.status("正在扫描文档并构建索引（首次运行会下载向量模型）..."):
        stats = store.build_index()
    console.print(f"\n[green bold]索引构建完成[/green bold]"
                  f"：{stats['docs']} 篇文档 → {stats['chunks']} 个片段，向量维度 {stats['dim']}\n")
    return 0


def cmd_ask(question: str, use_rerank: bool) -> int:
    """提问"""
    from src import rag
    with console.status("检索并生成回答中..."):
        result = rag.answer(question, use_rerank=use_rerank)

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
    console.print()
    return 0


def cmd_eval() -> int:
    """运行评测集"""
    from src import evaluate
    from src.retriever import HybridRetriever

    items = evaluate.load_eval_set()
    console.print(f"\n评测集：{len(items)} 条问题\n")

    # ---- 第一层：检索质量 ----
    with console.status("正在评测检索质量（Recall@K）..."):
        retriever = HybridRetriever()
        retrieval = evaluate.eval_retrieval(retriever, items)

    table = Table(title="1) 检索命中率（期望文档是否被召回）", show_lines=True)
    table.add_column("问题", overflow="fold", max_width=44)
    table.add_column("期望文档", style="cyan", no_wrap=True)
    table.add_column("命中", width=6, justify="center")
    table.add_column("排名", width=6, justify="center")
    for row in retrieval:
        table.add_row(row["question"], row["expected_doc"],
                      "[green]是[/green]" if row["hit"] else "[red]否[/red]",
                      str(row["rank"] or "-"))
    console.print(table)
    recall = sum(r["hit"] for r in retrieval) / len(retrieval)
    console.print(f"\nRecall@{len(retrieval[0]['top_docs']) or 5} = "
                  f"[bold]{recall:.0%}[/bold]（{sum(r['hit'] for r in retrieval)}/{len(retrieval)}）\n")

    # ---- 第二层：回答质量 ----
    with console.status("正在评测回答质量（LLM 裁判要点覆盖）..."):
        answers = evaluate.eval_answers(items, retriever)

    table2 = Table(title="2) 回答要点覆盖率（LLM 裁判）", show_lines=True)
    table2.add_column("问题", overflow="fold", max_width=44)
    table2.add_column("覆盖要点", overflow="fold", max_width=30)
    table2.add_column("覆盖分", width=8, justify="center")
    for row in answers:
        j = row["judge"]
        color = "green" if j["score"] >= 0.8 else ("yellow" if j["score"] >= 0.5 else "red")
        table2.add_row(row["question"],
                       "；".join(j["covered"]) or "-",
                       f"[{color}]{j['score']:.0%}[/{color}]")
    console.print(table2)

    avg = sum(r["judge"]["score"] for r in answers) / len(answers)
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
    sub.add_parser("ingest", help="构建知识库索引")
    ask_parser = sub.add_parser("ask", help="提问")
    ask_parser.add_argument("question", help="要问的问题")
    ask_parser.add_argument("--no-rerank", action="store_true", help="关闭重排（对比用）")
    sub.add_parser("eval", help="运行评测集")

    args = parser.parse_args()

    if args.check:
        return cmd_check()
    if args.command == "ingest":
        return cmd_ingest()
    if args.command == "ask":
        return cmd_ask(args.question, use_rerank=not args.no_rerank)
    if args.command == "eval":
        return cmd_eval()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
