"""core 层：抽象接口（契约）定义。

这里只放 ABC，不含任何具体实现，也不依赖第三方 SDK：
- llm_client   ：LLM 客户端
- vector_store ：向量库
- reranker     ：重排器

modules 层给出具体实现，services 层只依赖这里的接口做业务编排。
"""
from src.core.llm_client import BaseLLMClient
from src.core.reranker import BaseReranker
from src.core.vector_store import BaseVectorStore

__all__ = ["BaseLLMClient", "BaseVectorStore", "BaseReranker"]
