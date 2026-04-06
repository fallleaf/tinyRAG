#!/usr/bin/env python3
"""
chunker/markdown_splitter.py - 智能 Markdown 分块器 (v3.0)

v3.0 变更:
1. ✅ 置信度改为 Frontmatter 驱动 (doc_type / status / date)
2. ✅ 移除 path_rules (文件路径权重) 和 type_rules (内容块类型权重)
3. ✅ 新增日期指数衰减: weight = default * 2^(-days/half_life)
4. ✅ 线程安全: 局部变量 + 闭包，无实例级可变状态
5. ✅ 修正 max_chars: int(max_tokens * chars_per_token)
6. ✅ 修复代码块栅栏检测 Bug
7. ✅ 新增表格/列表块检测
8. ✅ @dataclass, 预编译正则，O(1) 缓冲区追踪
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from loguru import logger

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("⚠️ PyYAML 未安装，将使用简易 YAML 解析器 (功能受限)")


# ─── 预编译正则 ───
_RE_HEADER = re.compile(r"^(#{1,6})\s+(.*)")
_RE_FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})\s*(\w*)")
_RE_TABLE_ROW = re.compile(r"^\|")
_RE_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+\.)\s")
_RE_FRONTMATTER = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)", re.DOTALL)


class ChunkType(str, Enum):
    """分块类型枚举"""

    HEADER = "header"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    TEXT = "text"


@dataclass
class ChunkData:
    """分块数据实体"""

    file_id: int = 0
    chunk_index: int = 0
    content: str = ""
    content_type: ChunkType = ChunkType.TEXT
    section_title: str | None = None
    section_path: str = "Root"
    start_pos: int = 0
    end_pos: int = 0
    confidence_doc_type_weight: float = 1.0
    confidence_status_weight: float = 1.0
    confidence_date_weight: float = 1.0
    confidence_final_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.section_path, list):
            self.section_path = " / ".join(["Root", *self.section_path])
        elif not self.section_path:
            self.section_path = "Root"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items()}


class MarkdownSplitter:
    """智能 Markdown 分块器"""

    def __init__(
        self,
        max_tokens: int = 512,
        overlap: int = 50,
        confidence_config: dict[str, Any] | None = None,
        chars_per_token: float = 2.0,
    ):
        """
        Args:
            max_tokens: 嵌入模型最大 token 数
            overlap: 滑动窗口重叠字符数
            confidence_config: 置信度配置 (doc_type_rules / status_rules / date_decay)
            chars_per_token: 每个 token 对应的字符数
        """
        self.max_chars = int(max_tokens * chars_per_token)
        self.overlap = overlap

        # 加载 Frontmatter 置信度配置
        cc = confidence_config or {}
        self._doc_type_rules: dict[str, float] = cc.get("doc_type_rules", {})
        self._status_rules: dict[str, float] = cc.get("status_rules", {})
        self._default_weight: float = cc.get("default_weight", 1.0)

        date_cfg = cc.get("date_decay", {})
        self._date_decay_enabled: bool = date_cfg.get("enabled", True)
        self._date_half_life_days: int = date_cfg.get("half_life_days", 365)
        self._date_min_weight: float = date_cfg.get("min_weight", 0.5)

    # ─── Frontmatter 置信度计算 ───

    def _calc_doc_type_weight(self, metadata: dict[str, Any]) -> float:
        """根据 frontmatter doc_type 字段查表获取权重"""
        if not self._doc_type_rules:
            return 1.0
        doc_type = str(metadata.get("doc_type", "")).strip().lower()
        return self._doc_type_rules.get(doc_type, self._default_weight)

    def _calc_status_weight(self, metadata: dict[str, Any]) -> float:
        """根据 frontmatter status 字段查表获取权重"""
        if not self._status_rules:
            return 1.0
        status = str(metadata.get("status", "")).strip()
        if not status:
            return self._default_weight
        return self._status_rules.get(status, self._default_weight)

    def _calc_date_weight(self, metadata: dict[str, Any]) -> float:
        """
        根据日期计算时间衰减权重。
        公式: weight = default * 2^(-days_old / half_life_days)
        下限: min_weight
        """
        if not self._date_decay_enabled:
            return 1.0

        raw_date = metadata.get("date")
        if not raw_date:
            return self._default_weight

        try:
            # 兼容 date 对象、datetime 对象、字符串
            if isinstance(raw_date, (date, datetime)):
                doc_date = raw_date if isinstance(raw_date, date) else raw_date.date()
            else:
                doc_date = datetime.strptime(str(raw_date).strip()[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return self._default_weight

        days_old = (datetime.now().date() - doc_date).days
        if days_old <= 0:
            return self._default_weight

        # 指数衰减：每过 half_life_days 天，权重减半
        weight = self._default_weight * math.pow(2.0, -days_old / self._date_half_life_days)
        return max(weight, self._date_min_weight)

    def _calc_final_weight(self, metadata: dict[str, Any]) -> tuple[float, float, float, float]:
        """
        计算综合置信度权重。
        返回: (doc_type_weight, status_weight, date_weight, final_weight)
        """
        dt_w = self._calc_doc_type_weight(metadata)
        st_w = self._calc_status_weight(metadata)
        da_w = self._calc_date_weight(metadata)
        return dt_w, st_w, da_w, dt_w * st_w * da_w

    # ─── Frontmatter 提取 ───

    def extract_frontmatter(self, content: str) -> tuple[str, dict[str, Any], int]:
        """
        提取 YAML Frontmatter。
        返回: (剩余内容，元数据字典，frontmatter 占用的字符偏移量)
        """
        if not content.startswith("---"):
            return content, {}, 0

        m = _RE_FRONTMATTER.match(content)
        if not m:
            return content, {}, 0

        yaml_block = m.group(1)
        raw_remaining = content[m.end() :]
        leading_ws = len(raw_remaining) - len(raw_remaining.lstrip())
        fm_offset = m.end() + leading_ws
        remaining = raw_remaining.lstrip()

        try:
            if HAS_YAML:
                fm = yaml.safe_load(yaml_block)
                return remaining, fm if isinstance(fm, dict) else {}, fm_offset
            else:
                return remaining, self._parse_yaml_simple(yaml_block), fm_offset
        except Exception as e:
            logger.error(f"❌ YAML 解析失败:{e}")
            return remaining, {}, fm_offset

    def _parse_yaml_simple(self, yaml_content: str) -> dict[str, Any]:
        """简易 YAML 解析器 (PyYAML 不可用时的降级方案)"""
        result: dict[str, Any] = {}
        current_key: str | None = None
        current_list: list[str] | None = None

        for line in yaml_content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("- "):
                if current_list is not None:
                    item = line[2:].strip().strip("\"'")
                    current_list.append(item)
                continue

            if ":" in line:
                if current_key and current_list is not None:
                    result[current_key] = current_list
                    current_list = None

                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip().strip("\"'")

                if not value:
                    current_key, current_list = key, []
                else:
                    if value.lower() == "true":
                        result[key] = True
                    elif value.lower() == "false":
                        result[key] = False
                    else:
                        try:
                            result[key] = int(value)
                        except ValueError:
                            try:
                                result[key] = float(value)
                            except ValueError:
                                result[key] = value
                    current_key = None

        if current_key and current_list is not None:
            result[current_key] = current_list
        return result

    # ─── 分块核心 ───

    def split(
        self,
        content: str,
        file_path: str | None = None,
        file_id: int = 0,
    ) -> list[ChunkData]:
        """
        线程安全的分块方法。所有可变状态均为局部变量 / 闭包捕获。
        块类型识别优先级：标题 > 代码块 > 表格 > 列表 > 正文

        Args:
            content: Markdown 内容
            file_path: 文件路径 (未使用，保留用于未来扩展)
            file_id: 文件 ID
        """
        chunk_counter = [0]

        remaining_content, frontmatter, fm_offset = self.extract_frontmatter(content)
        lines = remaining_content.split("\n")
        chunks: list[ChunkData] = []

        # 预计算该文件的置信度权重 (同一文件所有 chunk 共享)
        dt_w, st_w, da_w, final_w = self._calc_final_weight(frontmatter)

        # ── 闭包：构建单个 ChunkData ──
        def make_chunk(text: str, c_type: ChunkType, stack: list[str], title: str, s_pos: int, e_pos: int) -> ChunkData:
            chunk = ChunkData(
                file_id=file_id,
                chunk_index=chunk_counter[0],
                content=text,
                content_type=c_type,
                section_title=title,
                section_path=list(stack),
                start_pos=s_pos + fm_offset,
                end_pos=e_pos + fm_offset,
                metadata=frontmatter,
                confidence_doc_type_weight=dt_w,
                confidence_status_weight=st_w,
                confidence_date_weight=da_w,
                confidence_final_weight=final_w,
            )
            chunk_counter[0] += 1
            return chunk

        # ── 闭包：带 overlap 的分块 (含空 chunk 过滤) ──
        def create_chunks(
            buf_lines: list[str], c_type: ChunkType, stack: list[str], start: int, end: int
        ) -> list[ChunkData]:
            full_text = "\n".join(buf_lines)
            if not full_text.strip():
                return []

            section_title = stack[-1] if stack else "Root"
            result: list[ChunkData] = []
            text_ptr = 0
            text_len = len(full_text)

            if text_len <= self.max_chars:
                result.append(
                    make_chunk(full_text, c_type, stack, section_title, start, start + text_len),
                )
                return result

            while text_ptr < text_len:
                end_ptr = min(text_ptr + self.max_chars, text_len)
                chunk_content = full_text[text_ptr:end_ptr]
                if end_ptr < text_len:
                    last_nl = chunk_content.rfind("\n")
                    if last_nl > self.max_chars * 0.7:
                        end_ptr = text_ptr + last_nl
                        chunk_content = full_text[text_ptr:end_ptr]

                stripped = chunk_content.strip()
                if stripped:
                    result.append(
                        make_chunk(stripped, c_type, stack, section_title, start + text_ptr, start + end_ptr),
                    )

                step = max(self.max_chars - self.overlap, 1)
                text_ptr += step
                if text_len - text_ptr <= self.overlap and text_ptr < text_len:
                    tail = full_text[text_ptr:].strip()
                    if tail:
                        result.append(
                            make_chunk(tail, c_type, stack, section_title, start + text_ptr, start + text_len),
                        )
                    break
            return result

        # ── 主解析循环 ──
        current_buffer: list[str] = []
        current_type = ChunkType.TEXT
        section_stack: list[str] = []
        start_pos = 0
        current_pos = 0
        buf_size = 0
        expected_fence: str | None = None

        for line in lines:
            line_len = len(line) + 1
            stripped = line.strip()

            # ── 1. 标题检测 ──
            header_match = _RE_HEADER.match(line)
            if header_match:
                if current_buffer:
                    chunks.extend(create_chunks(current_buffer, current_type, section_stack, start_pos, current_pos))
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                section_stack = [*section_stack[: level - 1], title]
                current_buffer = [line]
                current_type = ChunkType.HEADER
                start_pos = current_pos
                buf_size = line_len
                current_pos += line_len
                continue

            # ── 2. 代码块栅栏检测 ──
            fence_match = _RE_FENCE_OPEN.match(line)
            if fence_match:
                if current_type != ChunkType.CODE:
                    if current_buffer:
                        chunks.extend(
                            create_chunks(current_buffer, current_type, section_stack, start_pos, current_pos)
                        )
                    fence_str = fence_match.group(1)
                    expected_fence = fence_str[0] * len(fence_str)
                    current_buffer = [line]
                    current_type = ChunkType.CODE
                    start_pos = current_pos
                    buf_size = line_len
                elif expected_fence and stripped.startswith(expected_fence):
                    current_buffer.append(line)
                    buf_size += line_len
                    chunks.extend(
                        create_chunks(current_buffer, current_type, section_stack, start_pos, current_pos + line_len)
                    )
                    current_buffer = []
                    current_type = ChunkType.TEXT
                    start_pos = current_pos + line_len
                    buf_size = 0
                    expected_fence = None
                else:
                    current_buffer.append(line)
                    buf_size += line_len
                    current_pos += line_len
                continue

            # ── 3. 块类型检测 (表格 / 列表) ──
            is_table = bool(_RE_TABLE_ROW.match(stripped))
            is_list = bool(_RE_LIST_ITEM.match(line))
            is_blank = not stripped

            should_flush = False
            new_type: ChunkType | None = None

            if (current_type == ChunkType.TABLE and not is_table and not is_blank) or (
                current_type == ChunkType.LIST and not is_list and not is_blank
            ):
                should_flush = True
                new_type = ChunkType.TEXT
            elif current_type in (ChunkType.TEXT, ChunkType.HEADER):
                if is_table:
                    should_flush = bool(current_buffer)
                    new_type = ChunkType.TABLE
                elif is_list:
                    should_flush = bool(current_buffer)
                    new_type = ChunkType.LIST

            if should_flush and current_buffer:
                chunks.extend(create_chunks(current_buffer, current_type, section_stack, start_pos, current_pos))
                current_buffer = []
                buf_size = 0
                start_pos = current_pos

            if new_type:
                current_type = new_type

            # ── 4. 追加行 ──
            current_buffer.append(line)
            buf_size += line_len
            current_pos += line_len

            # ── 5. 超长保底切分 (不对 CODE 类型切分) ──
            if current_type != ChunkType.CODE and buf_size > self.max_chars * 2:
                chunks.extend(create_chunks(current_buffer, current_type, section_stack, start_pos, current_pos))
                current_buffer = []
                buf_size = 0
                start_pos = current_pos

        if current_buffer:
            chunks.extend(create_chunks(current_buffer, current_type, section_stack, start_pos, current_pos))

        return chunks
