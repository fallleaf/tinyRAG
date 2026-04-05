#!/usr/bin/env python3
# embedder/embed_engine.py
import time

from utils.logger import logger

from .model_factory import EmbeddingModel


class EmbeddingEngine:
    def __init__(  # ✅ 修复：双下划线
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

        logger.info(f"📊 开始向量化 {len(texts)} 条文本...")
        start_time = time.time()
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                batch_embs = self.model.get_embedding(batch)
                all_embeddings.extend(batch_embs)

                if (i + self.batch_size) % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.debug(
                        f"  已处理 {i + self.batch_size}/{len(texts)} 条 ({elapsed:.2f}s)"
                    )
            except Exception as e:
                # ✅ 修复：不再 continue，而是抛出异常以防止批次错位导致的数据损坏
                logger.error(f"❌ 批次 {i//self.batch_size} 处理失败: {e}")
                raise RuntimeError(f"向量化批次中断: {e}") from e

        elapsed = time.time() - start_time
        logger.success(f"✅ 向量化完成：{len(all_embeddings)} 条 ({elapsed:.2f}s)")
        return all_embeddings

    def get_dimension(self) -> int:
        return self.dimension
