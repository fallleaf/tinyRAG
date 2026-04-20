#!/usr/bin/env python3
# utils/jieba_helper.py
"""jieba 分词统一处理模块：日期保护、自定义词典加载"""

import re
from pathlib import Path

from utils.logger import logger

try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("⚠️ jieba 未安装，分词功能将降级")

_DATE_PATTERN = re.compile(r"\d{4}(?:-\d{2}(?:-\d{2})?|年(?:\d{1,2}(?:月(?:\d{1,2}日)?)?)?)")
_BROKEN_DATE_RE = re.compile(r"__\s*DATE\s*_\s*(\d+)\s*__")
_DOT_SPACING_RE = re.compile(r"\s*\.\s*")


def jieba_segment(text: str) -> str:
    """对中文文本进行 jieba 分词，保护日期格式免被拆分"""
    if not JIEBA_AVAILABLE or not text or not text.strip():
        return text.strip() if text else ""

    date_placeholders = {}
    protected_text = text
    for i, match in enumerate(_DATE_PATTERN.finditer(text)):
        placeholder = f"__DATE_{i}__"
        date_placeholders[placeholder] = match.group()
        protected_text = protected_text.replace(match.group(), placeholder, 1)

    segmented = " ".join(jieba.cut_for_search(protected_text))
    segmented = _BROKEN_DATE_RE.sub(r"__DATE_\1__", segmented)
    for ph, date_str in date_placeholders.items():
        segmented = segmented.replace(ph, date_str)
    return _DOT_SPACING_RE.sub(".", segmented)


def load_jieba_user_dict(config) -> None:
    """加载 jieba 自定义词典"""
    if not JIEBA_AVAILABLE:
        return
    if hasattr(config, "jieba_user_dict") and config.jieba_user_dict:
        dict_path = Path(config.jieba_user_dict).expanduser()
        if dict_path.exists():
            try:
                jieba.load_userdict(str(dict_path))
                logger.info(f"✅ jieba 自定义词典加载成功: {dict_path}")
            except Exception as e:
                logger.warning(f"⚠️ jieba 自定义词典加载失败: {e}")
        else:
            logger.warning(f"⚠️ jieba 自定义词典文件不存在: {dict_path}")
