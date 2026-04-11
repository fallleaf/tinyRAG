#!/usr/bin/env python3
"""
chunker/markdown_splitter.py - 智能 Markdown 分块器 (v4.1)

v4.1 修复内容:
1. ✅ 统一缺省值为英文：technical/completed（与 hybrid_engine 一致）
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
    logger.warning("⚠️ PyYAML 未安装，将使用简易 YAML 解析器")

_RE_HEADER = re.compile(r"^(#{1,6})\s+(.*)")
_RE_FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})\s*(\w*)")
_RE_TABLE_ROW = re.compile(r"^\|")
_RE_LIST_ITEM = re.compile(r"^(\s*)([*+-]|\d+\.)\s+")
_RE_YAML_BLOCK = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


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
    confidence_metadata: dict = field(default_factory=dict)


class MarkdownSplitter:
    def __init__(self, config: Any):
        self.config = config
        self.max_tokens = config.chunking.get("max_tokens", 512)
        self.overlap = config.chunking.get("overlap", 50)
        self.chars_per_token = 2.5
        self.max_chars = int(self.max_tokens * self.chars_per_token)
        self.overlap_chars = int(self.overlap * self.chars_per_token)

    def split(self, text: str, file_mtime: float | None) -> list[Chunk]:
        frontmatter, content_body = self._parse_frontmatter(text)
        conf_meta = self._extract_confidence_meta(frontmatter, file_mtime)

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
                    confidence_metadata=conf_meta,
                )
            ]

        return self._process_lines(content_body, create_chunks)

    def _extract_confidence_meta(self, frontmatter: dict, file_mtime: float | None) -> dict:
        """
        🔧 v4.1: 统一缺省值为英文
        - doc_type 缺省: technical
        - status 缺省: completed
        """
        # 1. 提取 doc_type (缺省: technical)
        doc_type = frontmatter.get("doc_type", "technical")

        # 2. 提取 status (缺省: completed)
        status = frontmatter.get("status", "completed")

        # 3. 提取 final_date (优先级: final_date > date > mtime > now)
        f_date_val = frontmatter.get("final_date") or frontmatter.get("date")

        if not f_date_val:
            if file_mtime:
                f_date_str = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
            else:
                f_date_str = datetime.now().strftime("%Y-%m-%d")
        else:
            if isinstance(f_date_val, (date, datetime)):
                f_date_str = f_date_val.strftime("%Y-%m-%d")
            else:
                f_date_str = str(f_date_val).split(" ")[0]

        return {"doc_type": doc_type, "status": status, "final_date": f_date_str}

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
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
                logger.warning(f"YAML 解析失败: {e}")

        data = {}
        for line in yaml_str.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip()] = val.strip().strip('"').strip("'")
        return data, content_body

    def _process_lines(self, text: str, create_chunks: Any) -> list[Chunk]:
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

            fence_match = _RE_FENCE_OPEN.match(stripped)
            if fence_match:
                if not in_code_block:
                    if current_buffer:
                        chunks.extend(
                            create_chunks(
                                current_buffer, current_type, section_stack, start_pos, current_pos
                            )
                        )
                    in_code_block = True
                    current_type = ChunkType.CODE
                    current_buffer = [line]
                    buf_size = line_len
                    start_pos = current_pos
                else:
                    current_buffer.append(line)
                    chunks.extend(
                        create_chunks(
                            current_buffer, ChunkType.CODE, section_stack, start_pos, current_pos + line_len
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

            header_match = _RE_HEADER.match(stripped)
            if header_match:
                if current_buffer:
                    chunks.extend(
                        create_chunks(
                            current_buffer, current_type, section_stack, start_pos, current_pos
                        )
                    )

                level = len(header_match.group(1))
                title = header_match.group(2)
                section_stack = section_stack[: level - 1]
                section_stack.append(title)

                current_type = ChunkType.HEADER
                current_buffer = [line]
                buf_size = line_len
                start_pos = current_pos
                current_pos += line_len
                continue

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
                        current_buffer, current_type, section_stack, start_pos, current_pos
                    )
                )
                current_buffer = []
                buf_size = 0
                start_pos = current_pos

            if new_type:
                current_type = new_type

            current_buffer.append(line)
            buf_size += line_len
            current_pos += line_len

            if buf_size > self.max_chars:
                chunks.extend(
                    create_chunks(
                        current_buffer, current_type, section_stack, start_pos, current_pos
                    )
                )
                if self.overlap_chars > 0 and current_type in (ChunkType.TEXT, ChunkType.HEADER):
                    overlap_lines = []
                    overlap_size = 0
                    for line in reversed(current_buffer):
                        if overlap_size + len(line) > self.overlap_chars:
                            break
                        overlap_lines.insert(0, line)
                        overlap_size += len(line)
                    current_buffer = overlap_lines
                    buf_size = overlap_size
                    start_pos = current_pos - overlap_size
                else:
                    current_buffer = []
                    buf_size = 0
                    start_pos = current_pos

        if current_buffer:
            chunks.extend(create_chunks(current_buffer, current_type, section_stack, start_pos, current_pos))

        return chunks
