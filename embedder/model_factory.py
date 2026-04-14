#!/usr/bin/env python3
# embedder/model_factory.py
import threading
import time

from utils.logger import logger

try:
    from fastembed import TextEmbedding
except ImportError as e:
    logger.error("❌ fastembed 未安装，请运行: pip install fastembed")
    raise ImportError("fastembed not found") from e


class EmbeddingModel:
    def __init__(self, model_name: str, cache_dir: str, unload_after_seconds: int = 30):  # ✅ 修复：双下划线
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.unload_after_seconds = unload_after_seconds
        self._model: TextEmbedding | None = None
        self._last_used: float = 0
        self._lock = threading.Lock()
        self._unload_timer: threading.Timer | None = None
        self.dimension: int = 0
        # 修复 M5: 在锁内调用 _init_model()，与 get_embedding() 中的加锁调用互斥
        with self._lock:
            self._init_model()

    def _init_model(self):
        try:
            logger.info(f"🔄 正在加载模型：{self.model_name} ...")
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
            self._last_used = time.time()
            logger.success(f"✅ 模型加载成功：{self.model_name}")

            dummy_vec = next(iter(self._model.embed(["test"])))
            self.dimension = len(dummy_vec)
            logger.info(f"📏 模型维度：{self.dimension}")
        except Exception as e:
            logger.critical(f"💥 模型加载失败：{e}")
            raise

    def _schedule_unload(self):
        if self._unload_timer:
            self._unload_timer.cancel()
        self._unload_timer = threading.Timer(self.unload_after_seconds, self.unload)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _unload_unlocked(self):
        """不加锁的卸载方法，由调用方负责持锁"""
        if self._model:
            logger.info("🗑️ 正在卸载模型以释放内存...")
            self._model = None
            self._last_used = 0
            if self._unload_timer:
                self._unload_timer.cancel()
                self._unload_timer = None
            logger.info("✅ 模型已卸载")

    def unload(self):
        """卸载模型（公开方法，自动加锁）"""
        with self._lock:
            self._unload_unlocked()

    def get_embedding(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._model is None:
                logger.info("🔄 模型已卸载，重新加载...")
                self._init_model()

            self._last_used = time.time()
            self._schedule_unload()

            try:
                # ✅ 修复：变量名拼写
                embeddings = list(self._model.embed(texts))
                return [list(e) for e in embeddings]
            except Exception as e:
                logger.error(f"❌ 向量化失败：{e}")
                # 修复 C1: 调用无锁版本，避免死锁
                self._unload_unlocked()
                raise
