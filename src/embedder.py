"""本地向量化：fastembed（ONNX 运行时，无需 torch / GPU）

首次运行会自动下载约 95MB 的 BGE 中文模型（国内自动走 hf-mirror 镜像），
下载一次后永久缓存在本地，之后完全离线可用。
"""
import numpy as np

_model = None


def _load_model():
    """惰性加载向量化模型（首次加载约 1~2 秒）"""
    global _model
    if _model is None:
        from . import config  # noqa: F401 —— 确保加载前 HF_ENDPOINT 等环境变量已就绪
        from fastembed import TextEmbedding

        _model = TextEmbedding(config.EMBEDDING_MODEL)
    return _model


def model_name() -> str:
    """当前使用的向量化模型名（用于日志与索引元信息）"""
    from . import config

    return config.EMBEDDING_MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """文本列表 → 归一化向量矩阵 (N, D)。归一化后内积即为余弦相似度"""
    model = _load_model()
    vectors = np.asarray(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_query(text: str) -> np.ndarray:
    """单条查询向量化"""
    return embed_texts([text])[0]
