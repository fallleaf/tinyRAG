#!/usr/bin/env python3
# utils/jieba_helper.py
"""jieba 分词统一处理模块：日期保护、自定义词典加载、分词模式控制

分词模式说明：
  - "search" (搜索引擎模式): jieba.cut_for_search，对长词自动拆分子词一并输出，
    例如 "中国联通" → "中国/国联/联通/中国联通"，召回率高但索引冗余
  - "precise" (精确模式, 默认): jieba.cut，只输出最优分词路径，
    例如 "中国联通" → "中国联通"，索引精简无冗余

对于 FTS5 全文检索，精确模式配合自定义词典即可获得准确的词语匹配；
语义召回由向量检索负责，无需依赖关键词子词拆分来提升召回。
"""
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
# 支持 jieba 将 DATE 拆分为 DA TE / D A T E 等格式，以及数字被拆分为 1 0 等
_BROKEN_DATE_RE = re.compile(r"__\s*D\s*A\s*T\s*E\s*_\s*([\d\s]+)\s*__")
_DOT_SPACING_RE = re.compile(r"\s*\.\s*")

# 分词模式：默认精确模式，避免搜索引擎模式下长词被拆分为冗余子词
_SEG_MODE = "precise"


def set_seg_mode(mode: str) -> None:
    """设置全局分词模式

    Args:
        mode: "precise" (精确模式, 默认) 或 "search" (搜索引擎模式)
    """
    global _SEG_MODE
    if mode not in ("precise", "search"):
        logger.warning(f"⚠️ 不支持的分词模式 '{mode}'，使用默认 'precise'")
        mode = "precise"
    _SEG_MODE = mode
    logger.info(f"✅ jieba 分词模式设置为: {mode}")


def get_seg_mode() -> str:
    """获取当前分词模式"""
    return _SEG_MODE


def _do_segment(text: str) -> str:
    """执行 jieba 分词，根据当前模式选择 cut 或 cut_for_search

    Args:
        text: 已经过日期占位符保护的文本

    Returns:
        空格分隔的分词结果字符串
    """
    if _SEG_MODE == "search":
        return " ".join(jieba.cut_for_search(text))
    else:
        return " ".join(jieba.cut(text))


def jieba_segment(text: str) -> str:
    """对中文文本进行 jieba 分词，保护日期格式免被拆分

    处理流程：
    1. 识别文本中的日期模式，替换为 __DATE_N__ 占位符
    2. 对保护后的文本执行 jieba 分词
    3. 修复 jieba 拆分占位符导致的 __ DA TE _ N __ 格式
    4. 将占位符还原为原始日期字符串
    5. 清理点号周围的冗余空格
    """
    if not JIEBA_AVAILABLE or not text or not text.strip():
        return text.strip() if text else ""

    date_placeholders = {}
    protected_text = text
    for i, match in enumerate(_DATE_PATTERN.finditer(text)):
        placeholder = f"__DATE_{i}__"
        date_placeholders[placeholder] = match.group()
        protected_text = protected_text.replace(match.group(), placeholder, 1)

    segmented = _do_segment(protected_text)
    segmented = _BROKEN_DATE_RE.sub(
        lambda m: f"__DATE_{re.sub(r'\s+', '', m.group(1))}__", segmented
    )
    for ph, date_str in date_placeholders.items():
        segmented = segmented.replace(ph, date_str)
    return _DOT_SPACING_RE.sub(".", segmented)


def load_jieba_user_dict(config) -> None:
    """加载 jieba 自定义词典，并从配置中读取分词模式

    自定义词典格式（每行）：
        词语 词频 词性
    例如：
        中国联通 1000 nz
        大语言模型 1000 n

    词频越高，jieba 越倾向于将该词作为整体切分；
    建议对专有名词设置较高词频（如 1000）以确保不被拆分。
    """
    if not JIEBA_AVAILABLE:
        return

    # 从配置中读取分词模式
    seg_mode = getattr(config, "jieba_seg_mode", "precise")
    set_seg_mode(seg_mode)

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
