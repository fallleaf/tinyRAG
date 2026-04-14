#!/usr/bin/env python3
# embedder/embed_engine.py - 批量向量化引擎 (P0/P1 修复版)
import time

from utils.logger import logger

from .model_factory import EmbeddingModel


class EmbeddingEngine:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        batch_size: int = 32,
        unload_after_seconds: int = 30,
    ):
        self.model = EmbeddingModel(model_name, cache_dir, unload_after_seconds)
        self.batch_size = batch_size
        self.dimension = self.model.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化"""
        if not texts:
            return []

        # 🔽 改为 DEBUG：避免与 tqdm 抢占终端，防止日志刷屏破坏进度条
        logger.debug(f"📊 开始向量化 {len(texts)} 条文本...")
        start_time = time.time()
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                batch_embs = self.model.get_embedding(batch)
                all_embeddings.extend(batch_embs)

                # 进度日志降级为 DEBUG，仅用于文件日志调试，不干扰终端动画
                if (i + self.batch_size) % 100 == 0 or i == 0:
                    elapsed = time.time() - start_time
                    logger.debug(f"  已处理 {i + self.batch_size}/{len(texts)} 条 ({elapsed:.2f}s)")
            except Exception as e:
                # 🔴 P0 修复：严格禁止 continue，防止批次错位导致向量与 Chunk 不匹配
                logger.error(f"❌ 批次 {i // self.batch_size} 处理失败: {e}")
                raise RuntimeError(f"向量化批次中断: {e}") from e

        elapsed = time.time() - start_time
        logger.debug(f"✅ 向量化完成：{len(all_embeddings)} 条 ({elapsed:.2f}s)")
        return all_embeddings

    def get_dimension(self) -> int:
        return self.dimension
