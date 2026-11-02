"""utils 层：通用工具（与业务无关，任何一层都可以直接用）。

当前暂无独立工具模块：重试语义内联在 modules/llm.py，日志统一用标准库 logging，
文本处理（切分 / JSON 抽取）分别内联在 modules/ingest.py 与 modules/rewrite.py。
后续若要抽出通用重试装饰器、结构化日志或 JSON 兜底解析，放这一层。
"""
