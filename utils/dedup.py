"""
Chunk 去重工具函数
用于检索后去重，避免 LLM 收到重复/高度相似内容
"""

import numpy as np
from typing import List, Tuple, Any


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        a: 向量 1
        b: 向量 2
    
    Returns:
        余弦相似度 [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)


def deduplicate_chunks(
    chunks_with_embeddings: List[Tuple[Any, np.ndarray]],
    threshold: float = 0.85
) -> List[Any]:
    """
    语义去重：移除相似度 > threshold 的 chunk

    Args:
        chunks_with_embeddings: [(chunk, embedding), ...]
        threshold: 相似度阈值，>threshold 视为重复

    Returns:
        去重后的 chunk 列表
    """
    if not chunks_with_embeddings:
        return []

    selected = []
    first_chunk, first_embedding = chunks_with_embeddings[0]
    selected.append((first_chunk, first_embedding))

    for chunk, embedding in chunks_with_embeddings[1:]:
        # 计算与已选 chunk 的最大相似度
        max_sim = max(
            cosine_sim(embedding, e)
            for _, e in selected
        )

        if max_sim < threshold:
            selected.append((chunk, embedding))

    return [chunk for chunk, _ in selected]


def deduplicate_with_scores(
    chunks_with_embeddings: List[Tuple[Any, np.ndarray]],
    threshold: float = 0.85
) -> List[Tuple[Any, float]]:
    """
    去重并保留相似度分数

    Args:
        chunks_with_embeddings: [(chunk, embedding, score), ...]
        threshold: 相似度阈值

    Returns:
        [(chunk, score), ...]
    """
    if not chunks_with_embeddings:
        return []

    selected = []
    first_item = chunks_with_embeddings[0]

    if len(first_item) == 2:
        chunk, embedding = first_item
        score = 1.0
    else:
        chunk, embedding, score = first_item

    selected.append((chunk, embedding, score))

    for item in chunks_with_embeddings[1:]:
        if len(item) == 2:
            chunk, embedding = item
            score = 1.0
        else:
            chunk, embedding, score = item

        # 计算与已选 chunk 的最大相似度
        max_sim = max(
            cosine_sim(embedding, e)
            for _, e, _ in selected
        )

        if max_sim < threshold:
            selected.append((chunk, embedding, score))

    return [(chunk, score) for chunk, _, score in selected]


def calculate_chunk_similarity_matrix(
    chunks_with_embeddings: List[Tuple[Any, np.ndarray]]
) -> np.ndarray:
    """
    计算所有 chunk 之间的相似度矩阵
    
    Args:
        chunks_with_embeddings: [(chunk, embedding), ...]
    
    Returns:
        相似度矩阵 (n x n)
    """
    n = len(chunks_with_embeddings)
    if n == 0:
        return np.array([]).reshape(0, 0)
    
    embeddings = np.array([e for _, e in chunks_with_embeddings])
    similarities = cosine_sim_matrix(embeddings)
    
    return similarities


def cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    批量计算余弦相似度矩阵
    
    Args:
        vectors: (n, d) 矩阵
    
    Returns:
        (n, n) 相似度矩阵
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-8)
    return np.dot(normalized, normalized.T)
