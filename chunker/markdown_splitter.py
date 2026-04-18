#!/usr/bin/env python3
"""
chunker/markdown_splitter.py - 智能 Markdown 分块器 (v4.0 - 检索期置信度重构)

重构说明:
1. ✅ 移除计算逻辑: 不再在分块阶段计算 final_weight。
2. ✅ 元数据提取: 仅提取 doc_type, status, final_date 原始字段。
3. ✅ 智能缺省值:
   - doc_type: 缺省为 "blog"
   - status: 缺省为 "已完成"
   - final_date: 优先级为 Frontmatter(final_date) > Frontmatter(date) > file_mtime > 运行当日。
4. ✅ 结构化输出: Chunk 对象新增 confidence_metadata 字段，便于 build_index 序列化。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from utils.logger import logger

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
_RE_LIST_ITEM = re.compile(r"^(\s*)([*+-]|\d+\.)\s+")
_RE_YAML_BLOCK = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# 句子边界正则（用于行内切分）
_RE_SENTENCE_BOUNDARY = re.compile(r"([。！？.!?]\s*|[；;]\s*|[\n])")


class ChunkType(Enum):
    TEXT = "text"
    HEADER = "header"
    CODE = "code"
    TABLE = "table"
    LIST = "list"


@dataclass
class Chunk:
    content: str
    content_type: ChunkType
    section_title: str | None
    section_path: str
    start_pos: int
    end_pos: int
    metadata: dict = field(default_factory=dict)
    # 核心新增：存储原始置信度因子，供检索期动态计算
    confidence_metadata: dict = field(default_factory=dict)


class MarkdownSplitter:
    def __init__(self, config: Any):
        self.config = config

        # 读取分块配置 (支持 ChunkingConfig 对象或 dict)
        chunking = config.chunking
        if hasattr(chunking, "max_tokens"):
            # Pydantic 模型
            self.max_tokens = chunking.max_tokens
            self.overlap = chunking.overlap
            self.token_mode = chunking.token_mode
            self.chinese_chars_per_token = chunking.chinese_chars_per_token
            self.english_chars_per_token = chunking.english_chars_per_token
            self.max_chars_multiplier = chunking.max_chars_multiplier
        else:
            # 兼容旧版 dict 配置
            self.max_tokens = chunking.get("max_tokens", 512)
            self.overlap = chunking.get("overlap", 50)
            self.token_mode = chunking.get("token_mode", "estimate")
            self.chinese_chars_per_token = chunking.get("chinese_chars_per_token", 1.5)
            self.english_chars_per_token = chunking.get("english_chars_per_token", 4.0)
            self.max_chars_multiplier = chunking.get("max_chars_multiplier", 2.5)

        # 尝试加载 tiktoken
        self._tiktoken_enc = None
        if self.token_mode == "tiktoken":
            try:
                import tiktoken

                self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
                logger.info("✅ tiktoken 已加载，使用精确 token 计算")
            except ImportError:
                logger.warning("⚠️ tiktoken 未安装，回退到估算模式")
                self.token_mode = "estimate"

        # 使用配置化的乘数计算保守上限
        self.max_chars = int(self.max_tokens * self.max_chars_multiplier)
        self.overlap_chars = int(self.overlap * 2.0)  # 重叠字符数

    def _count_tokens(self, text: str) -> int:
        """
        计算 token 数量

        模式:
        - tiktoken: 使用 tiktoken 库精确计算
        - estimate: 区分中英文估算 (参数可配置)
        """
        if self._tiktoken_enc:
            return len(self._tiktoken_enc.encode(text))

        # 估算模式：区分中英文 (使用配置化的参数)
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - chinese_chars

        # 使用配置化的字符/token 比例
        return int(chinese_chars / self.chinese_chars_per_token + other_chars / self.english_chars_per_token)

    def _estimate_chars_for_tokens(self, tokens: int, sample_text: str | None = None) -> int:
        """
        估算指定 token 数对应的字符数上限
        用于快速判断是否需要精确计算
        """
        # 保守估计：使用英文的 chars/token 比（最小比值，确保不会低估）
        return int(tokens * self.english_chars_per_token)

    def _split_long_content(
        self, text: str, c_type: ChunkType, s_stack: list[str], s_pos: int, frontmatter: dict, conf_meta: dict
    ) -> list[Chunk]:
        """
        对超长内容进行二次切分

        处理策略：
        1. TEXT/HEADER: 按句子边界切分
        2. CODE/TABLE/LIST:
           - 轻度超长（< 2x max_tokens）：保持完整
           - 极端超长（>= 2x max_tokens）：强制切分
        """
        s_title = s_stack[-1] if s_stack else None
        s_path = " / ".join(s_stack) if s_stack else "Root"

        token_count = self._count_tokens(text)

        # CODE/TABLE/LIST 类型：轻度超长时保持完整
        if c_type in (ChunkType.CODE, ChunkType.TABLE, ChunkType.LIST):
            if token_count < self.max_tokens * 2:
                # 轻度超长（< 2x），保持完整
                return [
                    Chunk(
                        content=text,
                        content_type=c_type,
                        section_title=s_title,
                        section_path=s_path,
                        start_pos=s_pos,
                        end_pos=s_pos + len(text),
                        metadata=frontmatter,
                        confidence_metadata=conf_meta,
                    )
                ]
            # 极端超长，需要切分
            logger.warning(f"⚠️ {c_type.value} 类型 chunk 严重超限 ({token_count} > {self.max_tokens * 2})，强制切分")

        # TEXT/HEADER 类型 或 极端超长的 CODE/TABLE/LIST：按句子边界切分
        chunks = []

        # 按句子边界分割
        parts = _RE_SENTENCE_BOUNDARY.split(text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentences.append(parts[i] + parts[i + 1])
            else:
                sentences.append(parts[i])
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1])

        if not sentences:
            sentences = [text]

        # 合并句子到接近 max_tokens
        current_text = ""
        current_start = s_pos

        for sentence in sentences:
            test_text = current_text + sentence
            test_tokens = self._count_tokens(test_text)

            # 如果单句就超过 max_tokens，需要强制切分
            if self._count_tokens(sentence) > self.max_tokens:
                # 先输出当前累积的内容
                if current_text.strip():
                    chunks.append(
                        Chunk(
                            content=current_text.strip(),
                            content_type=c_type,
                            section_title=s_title,
                            section_path=s_path,
                            start_pos=current_start,
                            end_pos=current_start + len(current_text),
                            metadata=frontmatter,
                            confidence_metadata=conf_meta,
                        )
                    )
                    current_start += len(current_text)

                # 按固定长度切分超长句子
                target_chars = int(self.max_tokens * self.chinese_chars_per_token * 0.9)

                for i in range(0, len(sentence), target_chars):
                    part = sentence[i : i + target_chars]
                    if part.strip():
                        chunks.append(
                            Chunk(
                                content=part.strip(),
                                content_type=c_type,
                                section_title=s_title,
                                section_path=s_path,
                                start_pos=current_start + i,
                                end_pos=current_start + i + len(part),
                                metadata=frontmatter,
                                confidence_metadata=conf_meta,
                            )
                        )

                current_text = ""
                current_start = (
                    s_pos + len(text[: text.find(sentence) + len(sentence)]) if sentence in text else current_start
                )

            elif test_tokens > self.max_tokens and current_text.strip():
                # 当前句子加入后会超限，先输出累积的内容
                chunks.append(
                    Chunk(
                        content=current_text.strip(),
                        content_type=c_type,
                        section_title=s_title,
                        section_path=s_path,
                        start_pos=current_start,
                        end_pos=current_start + len(current_text),
                        metadata=frontmatter,
                        confidence_metadata=conf_meta,
                    )
                )
                current_text = sentence
                current_start = s_pos + len(text[: text.find(sentence)]) if sentence in text else current_start
            else:
                current_text = test_text

        # 输出剩余内容
        if current_text.strip():
            chunks.append(
                Chunk(
                    content=current_text.strip(),
                    content_type=c_type,
                    section_title=s_title,
                    section_path=s_path,
                    start_pos=current_start,
                    end_pos=current_start + len(current_text),
                    metadata=frontmatter,
                    confidence_metadata=conf_meta,
                )
            )

        return (
            chunks
            if chunks
            else [
                Chunk(
                    content=text,
                    content_type=c_type,
                    section_title=s_title,
                    section_path=s_path,
                    start_pos=s_pos,
                    end_pos=s_pos + len(text),
                    metadata=frontmatter,
                    confidence_metadata=conf_meta,
                )
            ]
        )

    def split(self, text: str, file_mtime: float | None) -> list[Chunk]:
        """
        主分块函数：解析 Markdown 并提取置信度元数据
        """
        # 1. 提取 Frontmatter
        frontmatter, content_body = self._parse_frontmatter(text)

        # 2. 提取并补全置信度原始因子（注入缺省值）
        conf_meta = self._extract_confidence_meta(frontmatter, file_mtime)

        # 3. 准备分块闭包
        def create_chunks(
            lines: list[str],
            c_type: ChunkType,
            s_stack: list[str],
            s_pos: int,
            e_pos: int,
        ) -> list[Chunk]:
            chunk_text = "".join(lines).strip()
            if not chunk_text:
                return []

            # 检查 token 数是否超限
            token_count = self._count_tokens(chunk_text)
            if token_count > self.max_tokens:
                # 超限则进行二次切分
                return self._split_long_content(chunk_text, c_type, s_stack, s_pos, frontmatter, conf_meta)

            s_title = s_stack[-1] if s_stack else None
            s_path = " / ".join(s_stack) if s_stack else "Root"

            return [
                Chunk(
                    content=chunk_text,
                    content_type=c_type,
                    section_title=s_title,
                    section_path=s_path,
                    start_pos=s_pos,
                    end_pos=e_pos,
                    metadata=frontmatter,
                    confidence_metadata=conf_meta,  # 关键：将原始因子存入 chunk
                )
            ]

        return self._process_lines(content_body, create_chunks)

    def _extract_confidence_meta(self, frontmatter: dict, file_mtime: float | None) -> dict:
        """
        实现 1、2、5 点设想：提取原始因子并注入缺省值
        """
        # 1. 提取 doc_type (缺省: blog)
        doc_type = frontmatter.get("doc_type", "blog")

        # 2. 提取 status (缺省: 已完成)
        status = frontmatter.get("status", "已完成")

        # 3. 提取 final_date (优先级: final_date > date > mtime > now)
        f_date_val = frontmatter.get("final_date") or frontmatter.get("date")

        if not f_date_val:
            if file_mtime:
                f_date_str = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
            else:
                f_date_str = datetime.now().strftime("%Y-%m-%d")
        else:
            # 格式化日期确保为 YYYY-MM-DD 字符串
            if isinstance(f_date_val, (date, datetime)):
                f_date_str = f_date_val.strftime("%Y-%m-%d")
            else:
                # 兼容字符串中可能带有的时间部分
                f_date_str = str(f_date_val).split(" ")[0]

        return {"doc_type": doc_type, "status": status, "final_date": f_date_str}

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """解析 YAML Frontmatter，支持 PyYAML 或简易正则回退"""
        match = _RE_YAML_BLOCK.match(text)
        if not match:
            return {}, text

        yaml_str = match.group(1)
        content_body = text[match.end() :]

        if HAS_YAML:
            try:
                data = yaml.safe_load(yaml_str)
                if isinstance(data, dict):
                    return data, content_body
            except Exception as e:
                logger.warning(f"YAML 解析失败，切换到简易模式: {e}")

        # 简易正则解析逻辑 (作为 PyYAML 缺失或解析失败后的兜底)
        data = {}
        for line in yaml_str.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip()] = val.strip().strip('"').strip("'")
        return data, content_body

    def _process_lines(self, text: str, create_chunks: Any) -> list[Chunk]:
        """扫描行并根据 Markdown 语法规则进行智能切分 (状态机)"""
        chunks = []
        lines = text.splitlines(keepends=True)

        current_buffer = []
        buf_size = 0
        current_type = ChunkType.TEXT
        section_stack = []
        in_code_block = False
        start_pos = 0
        current_pos = 0

        for line in lines:
            line_len = len(line)
            stripped = line.strip()

            # ── 1. 代码块处理 (最高优先级) ──
            fence_match = _RE_FENCE_OPEN.match(stripped)
            if fence_match:
                if not in_code_block:
                    # 进入代码块前，刷出之前的 Buffer
                    if current_buffer:
                        chunks.extend(
                            create_chunks(
                                current_buffer,
                                current_type,
                                section_stack,
                                start_pos,
                                current_pos,
                            )
                        )
                    in_code_block = True
                    current_type = ChunkType.CODE
                    current_buffer = [line]
                    buf_size = line_len
                    start_pos = current_pos
                else:
                    # 退出代码块
                    current_buffer.append(line)
                    chunks.extend(
                        create_chunks(
                            current_buffer,
                            ChunkType.CODE,
                            section_stack,
                            start_pos,
                            current_pos + line_len,
                        )
                    )
                    in_code_block = False
                    current_buffer = []
                    buf_size = 0
                    current_type = ChunkType.TEXT
                    start_pos = current_pos + line_len

                current_pos += line_len
                continue

            if in_code_block:
                current_buffer.append(line)
                buf_size += line_len
                current_pos += line_len
                continue

            # ── 2. 标题处理 (更新层级) ──
            header_match = _RE_HEADER.match(stripped)
            if header_match:
                if current_buffer:
                    chunks.extend(
                        create_chunks(
                            current_buffer,
                            current_type,
                            section_stack,
                            start_pos,
                            current_pos,
                        )
                    )

                level = len(header_match.group(1))
                title = header_match.group(2)

                # 动态维护标题栈
                section_stack = section_stack[: level - 1]
                section_stack.append(title)

                current_type = ChunkType.HEADER
                current_buffer = [line]
                buf_size = line_len
                start_pos = current_pos
                current_pos += line_len
                continue

            # ── 3. 块类型探测 (表格、列表) ──
            is_table = bool(_RE_TABLE_ROW.match(stripped))
            is_list = bool(_RE_LIST_ITEM.match(line))
            is_blank = not stripped

            should_flush = False
            new_type = None

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
                chunks.extend(
                    create_chunks(
                        current_buffer,
                        current_type,
                        section_stack,
                        start_pos,
                        current_pos,
                    )
                )
                current_buffer = []
                buf_size = 0
                start_pos = current_pos

            if new_type:
                current_type = new_type

            # ── 4. 追加行 ──
            current_buffer.append(line)
            buf_size += line_len
            current_pos += line_len

            # ── 5. 超长切分 (TEXT/HEADER 类型带重叠) ──
            # 先用字符数快速判断，接近阈值时再用 token 精确计算
            should_split = False
            if buf_size > self.max_chars:
                # 已超过保守上限，必定切分
                should_split = True
            elif buf_size > self._estimate_chars_for_tokens(self.max_tokens * 0.8):
                # 接近阈值，精确计算 token
                buffer_text = "".join(current_buffer)
                if self._count_tokens(buffer_text) > self.max_tokens:
                    should_split = True

            if should_split:
                chunks.extend(
                    create_chunks(
                        current_buffer,
                        current_type,
                        section_stack,
                        start_pos,
                        current_pos,
                    )
                )
                # 仅对 TEXT/HEADER 类型实现重叠切分 (CODE/TABLE/LIST 保持完整语义边界)
                if self.overlap_chars > 0 and current_type in (
                    ChunkType.TEXT,
                    ChunkType.HEADER,
                ):
                    # 计算重叠 token 数对应的字符数
                    overlap_tokens = self.overlap
                    overlap_lines = []
                    overlap_text = ""

                    # 从后向前添加行，直到达到 overlap token 数
                    for line in reversed(current_buffer):
                        test_text = line + overlap_text
                        if self._count_tokens(test_text) > overlap_tokens:
                            break
                        overlap_lines.insert(0, line)
                        overlap_text = test_text

                    current_buffer = overlap_lines
                    buf_size = len(overlap_text)
                    start_pos = current_pos - buf_size
                else:
                    current_buffer = []
                    buf_size = 0
                    start_pos = current_pos

        # 刷出最后残留
        if current_buffer:
            chunks.extend(create_chunks(current_buffer, current_type, section_stack, start_pos, current_pos))

        return chunks
