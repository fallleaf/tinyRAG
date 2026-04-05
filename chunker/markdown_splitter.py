#!/usr/bin/env python3
"""
chunker/markdown_splitter.py - 智能 Markdown 分块器 (配置驱动加权版)
功能特性:
1. 动态置信度加权 (支持 config.yaml 的 path_rules / type_rules)
    2. 提取 Frontmatter (YAML) 元数据
    3. 多级标题追踪 (H1-H6) 构建完整路径
    4. 识别代码块/表格/列表,防止误切
    5. 滑动窗口 + Overlap 算法,保护语义完整性
"""

import re
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


class ChunkType(str, Enum):
    HEADER = "header"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    TEXT = "text"


class ChunkData:
    def __init__(
        self,
        file_id: int,
        chunk_index: int,
        content: str,
        content_type: ChunkType,
        section_title: str | None,
        section_path: Any,
        start_pos: int,
        end_pos: int,
        confidence_path_weight: float = 1.0,
        confidence_type_weight: float = 1.0,
        confidence_final_weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.file_id = file_id
        self.chunk_index = chunk_index
        self.content = content
        self.content_type = content_type
        self.section_title = section_title
        self.section_path = (
            " / ".join(["Root", *section_path]) if isinstance(section_path, list) else str(section_path or "Root")
        )
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.confidence_path_weight = confidence_path_weight
        self.confidence_type_weight = confidence_type_weight
        self.confidence_final_weight = confidence_final_weight
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class MarkdownSplitter:
    def __init__(
        self,
        max_tokens: int = 512,
        overlap: int = 50,
        confidence_config: dict[str, Any] | None = None,
    ):
        self.max_chars = max_tokens * 4
        self.overlap = overlap
        self.chunk_counter = 0
        self._current_source_path = ""

        # 加载置信度配置
        self.confidence_config = confidence_config or {}
        self._path_rules = self.confidence_config.get("path_rules", [])
        self._type_rules = self.confidence_config.get("type_rules", {})
        # fusion 参数通常用于检索阶段,此处仅作透传/预留
        self._fusion_config = self.confidence_config.get("fusion", {"alpha": 0.6, "beta": 0.2})

    def _calc_path_weight(self, source_path: str) -> float:
        if not self._path_rules or not source_path:
            return 1.0
        norm_path = PurePath(source_path).as_posix()
        for rule in self._path_rules:
            pattern = rule.get("pattern", "")
            # 支持 glob 风格匹配 (如 03.日记/**, **/*.md)
            if PurePath(norm_path).match(pattern):
                return rule.get("weight", 1.0)
        return 1.0

    def _calc_type_weight(self, c_type: ChunkType) -> float:
        return self._type_rules.get(c_type.value, 1.0)

    def extract_frontmatter(self, content: str) -> tuple[str, dict[str, Any]]:
        if not content.startswith("---"):
            return content, {}
        # 修复:支持末尾无换行/带空格/直接 EOF
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", content, re.DOTALL)
        if not frontmatter_match:
            return content, {}

        yaml_block = frontmatter_match.group(1)
        remaining_content = content[frontmatter_match.end() :].strip()
        try:
            if HAS_YAML:
                fm = yaml.safe_load(yaml_block)
                return remaining_content, fm if isinstance(fm, dict) else {}
            else:
                return remaining_content, self._parse_yaml_simple(yaml_block)
        except Exception as e:
            logger.error(f"❌ YAML 解析失败:{e}")
            return remaining_content, {}

    def _parse_yaml_simple(self, yaml_content: str) -> dict[str, Any]:
        result, current_key, current_list = {}, None, None
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

    def split(self, content: str, file_path: str | None = None, file_id: int = 0) -> list[ChunkData]:
        self.chunk_counter = 0
        self._current_source_path = file_path or ""  # ✅ 供置信度匹配使用

        remaining_content, frontmatter = self.extract_frontmatter(content)
        lines = remaining_content.split("\n")
        chunks = []
        current_buffer, current_type = [], ChunkType.TEXT
        section_stack: list[str] = []
        start_pos = current_pos = 0

        for line in lines:
            line_len = len(line) + 1
            # 1. 标题检测
            header_match = re.match(r"^(#{1,6})\s+(.*)", line)
            if header_match:
                if current_buffer:
                    chunks.extend(
                        self._create_chunks_with_overlap(
                            current_buffer,
                            current_type,
                            section_stack,
                            start_pos,
                            current_pos,
                            frontmatter,
                            file_id,
                        )
                    )
                level, title = len(header_match.group(1)), header_match.group(2).strip()
                section_stack = [*section_stack[: level - 1], title]
                current_buffer, current_type = [line], ChunkType.HEADER
                start_pos = current_pos
                current_pos += line_len
                continue

            # 2. 代码块检测 (支持 ```, ~~~ 及多反引号)
            fence_match = re.match(r"^(`{3,}|~{3,})\s*(\w+)?", line)
            if fence_match:
                fence_char, fence_len = fence_match.group(1)[0], len(fence_match.group(1))
                if current_type != ChunkType.CODE:
                    if current_buffer:
                        chunks.extend(
                            self._create_chunks_with_overlap(
                                current_buffer,
                                current_type,
                                section_stack,
                                start_pos,
                                current_pos,
                                frontmatter,
                                file_id,
                            )
                        )
                    current_buffer, current_type, start_pos = (
                        [line],
                        ChunkType.CODE,
                        current_pos,
                    )
                elif line.strip().startswith(fence_char * fence_len):
                    current_buffer.append(line)
                    chunks.extend(
                        self._create_chunks_with_overlap(
                            current_buffer,
                            current_type,
                            section_stack,
                            start_pos,
                            current_pos + line_len,
                            frontmatter,
                            file_id,
                        )
                    )
                    current_buffer, current_type, start_pos = (
                        [],
                        ChunkType.TEXT,
                        current_pos + line_len,
                    )
                current_pos += line_len
                continue

            current_buffer.append(line)
            current_pos += line_len
            # 3. 超长段落保底切分
            if sum(len(line) for line in current_buffer) > self.max_chars * 2:
                chunks.extend(
                    self._create_chunks_with_overlap(
                        current_buffer,
                        current_type,
                        section_stack,
                        start_pos,
                        current_pos,
                        frontmatter,
                        file_id,
                    )
                )
                current_buffer, start_pos = [], current_pos

        if current_buffer:
            chunks.extend(
                self._create_chunks_with_overlap(
                    current_buffer,
                    current_type,
                    section_stack,
                    start_pos,
                    current_pos,
                    frontmatter,
                    file_id,
                )
            )
        return chunks

    def _create_chunks_with_overlap(self, lines, c_type, stack, start, end, meta, f_id):
        full_text = "\n".join(lines)
        section_title = stack[-1] if stack else "Root"
        result, text_ptr, text_len = [], 0, len(full_text)

        if text_len <= self.max_chars:
            result.append(
                self._build_chunk_obj(
                    full_text,
                    c_type,
                    stack,
                    section_title,
                    start,
                    start + text_len,
                    meta,
                    f_id,
                )
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

            result.append(
                self._build_chunk_obj(
                    chunk_content.strip(),
                    c_type,
                    stack,
                    section_title,
                    start + text_ptr,
                    start + end_ptr,
                    meta,
                    f_id,
                )
            )

            # 修复:确定性步长 + 尾部安全处理
            step = max(self.max_chars - self.overlap, 1)
            text_ptr += step
            if text_len - text_ptr <= self.overlap and text_ptr < text_len:
                tail = full_text[text_ptr:].strip()
                if tail:
                    result.append(
                        self._build_chunk_obj(
                            tail,
                            c_type,
                            stack,
                            section_title,
                            start + text_ptr,
                            start + text_len,
                            meta,
                            f_id,
                        )
                    )
                break
        return result

    def _build_chunk_obj(self, content, c_type, stack, title, s_pos, e_pos, meta, f_id):
        path_w = self._calc_path_weight(self._current_source_path)
        type_w = self._calc_type_weight(c_type)
        final_w = path_w * type_w  # 默认乘法融合,可按需改为加权平均

        chunk = ChunkData(
            file_id=f_id,
            chunk_index=self.chunk_counter,
            content=content,
            content_type=c_type,
            section_title=title,
            section_path=list(stack),
            start_pos=s_pos,
            end_pos=e_pos,
            metadata=meta,
            confidence_path_weight=path_w,
            confidence_type_weight=type_w,
            confidence_final_weight=final_w,
        )
        self.chunk_counter += 1
        return chunk
