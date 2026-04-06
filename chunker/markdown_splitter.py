#!/usr/bin/env python3
"""
chunker/markdown_splitter.py - 智能 Markdown 分块器 (v2.0)
v2.0 优化:
1. ✅ 线程安全: 移除实例级可变状态 (chunk_counter, _current_source_path)
2. ✅ 修正 max_chars 计算: chars_per_token 参数适配中文场景
3. ✅ 修复代码块栅栏检测 Bug (内部栅栏行不再丢失)
4. ✅ 新增表格块检测 (管道符分隔行 → ChunkType.TABLE)
5. ✅ 新增列表块检测 (无序/有序列表 → ChunkType.LIST)
6. ✅ 修正位置追踪 (相对原始文件内容, 含 frontmatter 偏移)
7. ✅ ChunkData 改用 @dataclass, 修正类型注解
8. ✅ 过滤空 chunk, 编译正则, 缓冲区大小 O(1) 追踪
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePath
from typing import Any

from loguru import logger

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("⚠️ PyYAML 未安装,将使用简易 YAML 解析器 (功能受限)")


# ─── 预编译正则 (避免循环内重复编译) ───
_RE_HEADER = re.compile(r"^(#{1,6})\s+(.*)")
_RE_FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})\s*(\w*)")
_RE_TABLE_ROW = re.compile(r"^\|")
_RE_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+\.)\s")
_RE_FRONTMATTER = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)", re.DOTALL)


class ChunkType(str, Enum):
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
    confidence_path_weight: float = 1.0
    confidence_type_weight: float = 1.0
    confidence_final_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """将 list 形式的 section_path 转为 "Root / H1 / H2" 字符串"""
        if isinstance(self.section_path, list):
            self.section_path = " / ".join(["Root", *self.section_path])
        elif not self.section_path:
            self.section_path = "Root"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class MarkdownSplitter:
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
            confidence_config: 置信度配置 (path_rules / type_rules)
            chars_per_token: 每个 token 对应的字符数
                英文 ≈ 4.0, 中文 ≈ 1.5-2.0, 混合内容建议 2.0-2.5
        """
        self.max_chars = int(max_tokens * chars_per_token)
        self.overlap = overlap

        # 加载置信度配置
        self.confidence_config = confidence_config or {}
        self._path_rules = self.confidence_config.get("path_rules", [])
        self._type_rules = self.confidence_config.get("type_rules", {})

    def _calc_path_weight(self, source_path: str) -> float:
        if not self._path_rules or not source_path:
            return 1.0
        norm_path = PurePath(source_path).as_posix()
        for rule in self._path_rules:
            pattern = rule.get("pattern", "")
            if PurePath(norm_path).match(pattern):
                return rule.get("weight", 1.0)
        return 1.0

    def _calc_type_weight(self, c_type: ChunkType) -> float:
        return self._type_rules.get(c_type.value, 1.0)

    def extract_frontmatter(self, content: str) -> tuple[str, dict[str, Any], int]:
        """
        提取 YAML Frontmatter。
        返回: (剩余内容, 元数据字典, frontmatter 占用的字符偏移量)
        偏移量用于修正 start_pos/end_pos 使其指向原始文件位置。
        """
        if not content.startswith("---"):
            return content, {}, 0

        m = _RE_FRONTMATTER.match(content)
        if not m:
            return content, {}, 0

        yaml_block = m.group(1)
        raw_remaining = content[m.end():]
        # 计算实际偏移: frontmatter 结束位置 + 前导空白
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

    def split(
        self, content: str, file_path: str | None = None, file_id: int = 0
    ) -> list[ChunkData]:
        """
        线程安全的分块方法。
        所有可变状态均为局部变量 / 闭包捕获, 支持多线程并发调用。

        块类型识别优先级: 标题 > 代码块 > 表格 > 列表 > 正文
        """
        # ── 局部状态 (线程安全) ──
        source_path = file_path or ""
        chunk_counter = [0]  # 可变计数器, 供闭包内递增

        remaining_content, frontmatter, fm_offset = self.extract_frontmatter(content)
        lines = remaining_content.split("\n")
        chunks: list[ChunkData] = []

        # ── 闭包: 构建单个 ChunkData ──
        def make_chunk(text: str, c_type: ChunkType, stack: list[str],
                       title: str, s_pos: int, e_pos: int) -> ChunkData:
            path_w = self._calc_path_weight(source_path)
            type_w = self._calc_type_weight(c_type)
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
                confidence_path_weight=path_w,
                confidence_type_weight=type_w,
                confidence_final_weight=path_w * type_w,
            )
            chunk_counter[0] += 1
            return chunk

        # ── 闭包: 带 overlap 的分块 (含空 chunk 过滤) ──
        def create_chunks(buf_lines: list[str], c_type: ChunkType,
                          stack: list[str], start: int, end: int) -> list[ChunkData]:
            full_text = "\n".join(buf_lines)
            if not full_text.strip():
                return []

            section_title = stack[-1] if stack else "Root"
            result: list[ChunkData] = []
            text_ptr = 0
            text_len = len(full_text)

            if text_len <= self.max_chars:
                result.append(
                    make_chunk(full_text, c_type, stack, section_title, start, start + text_len)
                )
                return result

            while text_ptr < text_len:
                end_ptr = min(text_ptr + self.max_chars, text_len)
                chunk_content = full_text[text_ptr:end_ptr]
                # 优先在换行处切分, 避免切断行
                if end_ptr < text_len:
                    last_nl = chunk_content.rfind("\n")
                    if last_nl > self.max_chars * 0.7:
                        end_ptr = text_ptr + last_nl
                        chunk_content = full_text[text_ptr:end_ptr]

                stripped = chunk_content.strip()
                if stripped:
                    result.append(
                        make_chunk(stripped, c_type, stack, section_title,
                                   start + text_ptr, start + end_ptr)
                    )

                step = max(self.max_chars - self.overlap, 1)
                text_ptr += step
                # 尾部安全处理: 剩余内容 <= overlap 时合并到尾部 chunk
                if text_len - text_ptr <= self.overlap and text_ptr < text_len:
                    tail = full_text[text_ptr:].strip()
                    if tail:
                        result.append(
                            make_chunk(tail, c_type, stack, section_title,
                                       start + text_ptr, start + text_len)
                        )
                    break
            return result

        # ── 主解析循环 ──
        current_buffer: list[str] = []
        current_type = ChunkType.TEXT
        section_stack: list[str] = []
        start_pos = 0
        current_pos = 0
        buf_size = 0  # O(1) 缓冲区字符数追踪
        expected_fence: str | None = None  # 期望的代码块关闭栅栏

        for line in lines:
            line_len = len(line) + 1  # +1 补偿 split("\n") 丢失的换行符
            stripped = line.strip()

            # ── 1. 标题检测 (最高优先级) ──
            header_match = _RE_HEADER.match(line)
            if header_match:
                if current_buffer:
                    chunks.extend(create_chunks(current_buffer, current_type,
                                                section_stack, start_pos, current_pos))
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
                    # 开启代码块
                    if current_buffer:
                        chunks.extend(create_chunks(current_buffer, current_type,
                                                    section_stack, start_pos, current_pos))
                    fence_str = fence_match.group(1)
                    expected_fence = fence_str[0] * len(fence_str)
                    current_buffer = [line]
                    current_type = ChunkType.CODE
                    start_pos = current_pos
                    buf_size = line_len
                else:
                    # 在代码块内部: 判断是否为关闭栅栏
                    if expected_fence and stripped.startswith(expected_fence):
                        # 关闭代码块
                        current_buffer.append(line)
                        buf_size += line_len
                        chunks.extend(create_chunks(current_buffer, current_type,
                                                    section_stack, start_pos,
                                                    current_pos + line_len))
                        current_buffer = []
                        current_type = ChunkType.TEXT
                        start_pos = current_pos + line_len
                        buf_size = 0
                        expected_fence = None
                    else:
                        # 代码块内匹配栅栏模式但非关闭行 → 追加为内容 (不丢弃!)
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

            # TABLE/LIST 结束条件: 遇到非自身类型的非空行
            if current_type == ChunkType.TABLE and not is_table and not is_blank:
                should_flush = True
                new_type = ChunkType.TEXT
            elif current_type == ChunkType.LIST and not is_list and not is_blank:
                should_flush = True
                new_type = ChunkType.TEXT
            # TEXT/HEADER → TABLE/LIST 开始
            elif current_type in (ChunkType.TEXT, ChunkType.HEADER):
                if is_table:
                    should_flush = bool(current_buffer)
                    new_type = ChunkType.TABLE
                elif is_list:
                    should_flush = bool(current_buffer)
                    new_type = ChunkType.LIST

            if should_flush and current_buffer:
                chunks.extend(create_chunks(current_buffer, current_type,
                                            section_stack, start_pos, current_pos))
                current_buffer = []
                buf_size = 0
                start_pos = current_pos

            if new_type:
                current_type = new_type

            # ── 4. 追加行 ──
            current_buffer.append(line)
            buf_size += line_len
            current_pos += line_len

            # ── 5. 超长保底切分 (不对 CODE 类型切分, 保护代码块完整性) ──
            if current_type != ChunkType.CODE and buf_size > self.max_chars * 2:
                chunks.extend(create_chunks(current_buffer, current_type,
                                            section_stack, start_pos, current_pos))
                current_buffer = []
                buf_size = 0
                start_pos = current_pos

        # ── 处理尾部缓冲区 ──
        if current_buffer:
            chunks.extend(create_chunks(current_buffer, current_type,
                                        section_stack, start_pos, current_pos))

        return chunks
