#!/usr/bin/env python3
"""
extractor.py - 双层实体/关系抽取管道

实现文档级（Frontmatter/Wikilinks）和 Chunk 级（NLP/LLM）的实体/关系抽取。
"""
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field

from plugins.tinyrag_memory_graph.config import ExtractionConfig, MemoryGraphConfig
from plugins.tinyrag_memory_graph.models import Entity, EntityType, Relation, RelationType


@dataclass
class ExtractionResult:
    """抽取结果"""
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)
    processing_time_ms: float = 0.0
    source: str = "unknown"


class FrontmatterParser:
    """
    YAML Frontmatter 解析器

    提取文档级元数据，包括：
    - tags: 标签列表
    - related: 关联文档
    - author: 作者
    - project: 项目
    - status: 状态
    - aliases: 别名
    - 其他自定义字段
    """

    # YAML Frontmatter 边界正则
    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n',
        re.DOTALL | re.MULTILINE
    )

    @classmethod
    def parse(cls, content: str) -> dict:
        """解析 Frontmatter"""
        match = cls.FRONTMATTER_PATTERN.match(content)
        if not match:
            return {}

        yaml_content = match.group(1)
        try:
            import yaml
            return yaml.safe_load(yaml_content) or {}
        except Exception:
            # 简单解析作为后备
            return cls._simple_parse(yaml_content)

    @classmethod
    def _simple_parse(cls, yaml_content: str) -> dict:
        """简单 YAML 解析（处理常见格式）"""
        result = {}
        for line in yaml_content.split('\n'):
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()

            # 处理列表值
            if value.startswith('[') and value.endswith(']'):
                value = [v.strip() for v in value[1:-1].split(',') if v.strip()]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            result[key] = value

        return result

    @classmethod
    def extract_frontmatter_fields(cls, frontmatter: dict) -> dict:
        """提取标准化的 Frontmatter 字段"""
        return {
            "tags": cls._ensure_list(frontmatter.get("tags", [])),
            "related": cls._ensure_list(frontmatter.get("related", [])),
            "author": frontmatter.get("author", ""),
            "project": frontmatter.get("project", ""),
            "status": frontmatter.get("status", ""),
            "aliases": cls._ensure_list(frontmatter.get("aliases", [])),
            "doc_type": frontmatter.get("doc_type", frontmatter.get("type", "")),
            "created": frontmatter.get("created", frontmatter.get("date", "")),
            "title": frontmatter.get("title", ""),
        }

    @classmethod
    def _ensure_list(cls, value) -> list:
        """确保值为列表"""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(',') if v.strip()]
        return []


class WikilinkExtractor:
    """
    Obsidian Wikilink 提取器

    提取 [[Wikilink]] 格式的内部链接，支持：
    - [[链接文本]]
    - [[链接文本|显示文本]]
    - [[链接文本#标题]]
    - [[链接文本#标题|显示文本]]
    """

    # Wikilink 正则模式
    WIKILINK_PATTERN = re.compile(
        r'\[\[([^\]|#]+)(?:[#|]([^\]]*))?\]\]',
        re.MULTILINE
    )

    @classmethod
    def extract(cls, content: str) -> list[str]:
        """提取所有 Wikilink 目标"""
        links = []
        for match in cls.WIKILINK_PATTERN.finditer(content):
            target = match.group(1).strip()
            if target:
                links.append(target)
        return list(set(links))  # 去重

    @classmethod
    def extract_with_positions(cls, content: str) -> list[dict]:
        """提取 Wikilink 及其位置"""
        results = []
        for match in cls.WIKILINK_PATTERN.finditer(content):
            target = match.group(1).strip()
            if target:
                results.append({
                    "target": target,
                    "start": match.start(),
                    "end": match.end(),
                    "heading": match.group(2).strip() if match.group(2) else None,
                })
        return results


class NLPExtractor:
    """
    基于 spaCy 的 NLP 实体抽取器

    支持中文和英文实体识别。
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._nlp = None
        self._initialized = False

    def _ensure_nlp(self):
        """延迟加载 spaCy 模型"""
        if self._initialized:
            return

        try:
            import spacy
            model_name = self.config.spacy_model

            # 尝试加载模型
            try:
                self._nlp = spacy.load(model_name)
            except OSError:
                # 尝试下载模型
                from spacy.cli import download
                download(model_name)
                self._nlp = spacy.load(model_name)

            self._initialized = True
        except Exception as e:
            print(f"[NLPExtractor] Failed to load spaCy model: {e}")
            self._nlp = None
            self._initialized = True

    def extract_entities(self, text: str) -> list[Entity]:
        """从文本中抽取实体"""
        self._ensure_nlp()

        if not self._nlp:
            return []

        entities = []
        try:
            doc = self._nlp(text[:5000])  # 限制长度防止超时

            for ent in doc.ents:
                entity = Entity(
                    id=self._generate_entity_id(ent.text, ent.label_),
                    canonical_name=ent.text.strip(),
                    type=self._map_entity_type(ent.label_),
                    confidence=min(1.0, ent._.get("confidence", 0.8)) if hasattr(ent, "_") else 0.8,
                    source="spacy"
                )
                entities.append(entity)
        except Exception as e:
            print(f"[NLPExtractor] Error extracting entities: {e}")

        return self._deduplicate_entities(entities)

    def _map_entity_type(self, spacy_label: str) -> str:
        """映射 spaCy 实体类型到标准类型"""
        mapping = {
            # 英文
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORG,
            "GPE": EntityType.LOC,
            "LOC": EntityType.LOC,
            "DATE": EntityType.DATE,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.TECHNOLOGY,
            "WORK_OF_ART": EntityType.CONCEPT,
            # 中文 (ORG, LOC, DATE 与英文相同，不重复)
            "PER": EntityType.PERSON,
            "TIME": EntityType.DATE,
        }
        return mapping.get(spacy_label, EntityType.MISC)

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """生成实体 ID"""
        normalized = f"{entity_type}:{name}".lower()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """去重实体"""
        seen = {}
        for e in entities:
            if e.id not in seen or e.confidence > seen[e.id].confidence:
                seen[e.id] = e
        return list(seen.values())


class RuleExtractor:
    """
    基于规则的实体抽取器

    使用预定义词典和规则进行实体识别。
    """

    # 默认规则词典
    DEFAULT_PATTERNS = {
        # 项目模式
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+项目)\b': EntityType.PROJECT,
        # 技术术语
        r'\b([A-Z]{2,}|(?:Python|Java|JavaScript|TypeScript|Rust|Go|React|Vue|Node\.js))\b': EntityType.TECHNOLOGY,
        # 版本号
        r'\b(v?\d+\.\d+(?:\.\d+)?)\b': EntityType.CONCEPT,
        # 邮箱
        r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b': EntityType.PERSON,
        # URL
        r'(https?://[^\s]+)': EntityType.CONCEPT,
    }

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.patterns = self.DEFAULT_PATTERNS.copy()
        self._load_custom_dict()

    def _load_custom_dict(self):
        """加载自定义词典"""
        if self.config.rule_dict_path:
            try:
                import json
                with open(self.config.rule_dict_path, encoding="utf-8") as f:
                    custom = json.load(f)
                    for pattern, entity_type in custom.get("patterns", {}).items():
                        self.patterns[pattern] = entity_type
            except Exception as e:
                print(f"[RuleExtractor] Failed to load custom dict: {e}")

    def extract_entities(self, text: str) -> list[Entity]:
        """从文本中抽取实体"""
        entities = []

        for pattern, entity_type in self.patterns.items():
            try:
                for match in re.finditer(pattern, text):
                    name = match.group(1) if match.lastindex else match.group(0)
                    entity = Entity(
                        id=hashlib.md5(f"{entity_type}:{name}".encode()).hexdigest()[:16],
                        canonical_name=name.strip(),
                        type=entity_type,
                        confidence=1.0,  # 规则匹配置信度高
                        source="rule"
                    )
                    entities.append(entity)
            except Exception:
                continue

        return self._deduplicate_entities(entities)

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """去重实体"""
        seen = {}
        for e in entities:
            if e.id not in seen:
                seen[e.id] = e
        return list(seen.values())


class LLMExtractor:
    """
    基于 LLM 的实体/关系抽取器

    用于处理复杂文本或 NLP 失败时的后备方案。
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._llm = None

    def extract_entities(self, text: str) -> tuple[list[Entity], list[Relation]]:
        """使用 LLM 抽取实体和关系"""
        # 检查是否有可用的 LLM
        try:
            import llama_cpp
            if self._llm is None:
                # 尝试加载量化模型
                model_path = self._find_llm_model()
                if model_path:
                    self._llm = llama_cpp.Llama(
                        model_path=str(model_path),
                        n_ctx=2048,
                        n_threads=4,
                        verbose=False
                    )
        except ImportError:
            return [], []

        if not self._llm:
            return [], []

        # 构造提示词
        prompt = f"""请从以下文本中抽取实体和关系，以 JSON 格式返回。

文本：
{text[:2000]}

请返回 JSON 格式：
{{"entities": [{{"name": "实体名", "type": "类型"}}], "relations": [{{"src": "实体1", "tgt": "实体2", "type": "关系类型"}}]}}

JSON:"""

        try:
            # 设置超时
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("LLM extraction timeout")

            # 仅在 Unix 系统上使用 signal
            import os
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.config.llm_max_latency_ms // 1000 + 1)

            try:
                response = self._llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.1,
                    stop=["}"]
                )
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)

            # 解析响应
            text_response = response['choices'][0]['text']
            json_str = text_response + "}"  # 补全 JSON

            # 尝试找到 JSON 部分
            import json
            start = json_str.find('{')
            if start >= 0:
                data = json.loads(json_str[start:])

                entities = []
                for e in data.get("entities", []):
                    entities.append(Entity(
                        id=hashlib.md5(f"{e['type']}:{e['name']}".encode()).hexdigest()[:16],
                        canonical_name=e["name"],
                        type=e.get("type", "UNKNOWN"),
                        confidence=0.7,
                        source="llm"
                    ))

                return entities, []  # 关系抽取需要更复杂的处理

        except (TimeoutError, json.JSONDecodeError, Exception) as e:
            print(f"[LLMExtractor] Error: {e}")

        return [], []

    def _find_llm_model(self):
        """查找 LLM 模型文件"""
        candidates = [
            "~/.cache/llama.cpp/qwen1.5-0.5b-chat-q4_k_m.gguf",
            "~/.cache/llama.cpp/qwen-1.8b-chat-q4_k_m.gguf",
            "/models/llm/*.gguf",
        ]

        from pathlib import Path
        for pattern in candidates:
            expanded = Path(pattern).expanduser()
            if expanded.exists():
                return expanded
            # 尝试 glob
            parent = expanded.parent
            if parent.exists():
                matches = list(parent.glob(expanded.name.replace("*", "*")))
                if matches:
                    return matches[0]

        return None


class DualLayerExtractor:
    """
    双层抽取管道

    协调文档级和 Chunk 级抽取，实现 FR-1.2/FR-1.3 需求。
    """

    def __init__(self, config: MemoryGraphConfig):
        self.config = config
        self.extraction_config = config.extraction

        # 初始化抽取器
        self.nlp_extractor = NLPExtractor(self.extraction_config) if self.extraction_config.chunk_mode == "spacy" else None
        self.rule_extractor = RuleExtractor(self.extraction_config)
        self.llm_extractor = LLMExtractor(self.extraction_config) if self.extraction_config.chunk_mode == "llm_fallback" else None

    def extract_document_level(self, content: str, note_id: str) -> ExtractionResult:
        """
        文档级抽取（FR-1.2）

        解析 YAML Frontmatter + [[Wikilink]]
        """
        start_time = time.time()

        result = ExtractionResult(source="document_level")

        # 1. 解析 Frontmatter
        frontmatter = FrontmatterParser.parse(content)
        result.frontmatter = FrontmatterParser.extract_frontmatter_fields(frontmatter)

        # 2. 提取 Wikilinks
        wikilinks = WikilinkExtractor.extract(content)
        result.wikilinks = wikilinks

        # 3. 从 Frontmatter 创建实体
        if result.frontmatter.get("author"):
            result.entities.append(Entity(
                id=hashlib.md5(f"person:{result.frontmatter['author']}".encode()).hexdigest()[:16],
                canonical_name=result.frontmatter["author"],
                type=EntityType.PERSON,
                confidence=1.0,
                source="frontmatter"
            ))

        if result.frontmatter.get("project"):
            result.entities.append(Entity(
                id=hashlib.md5(f"project:{result.frontmatter['project']}".encode()).hexdigest()[:16],
                canonical_name=result.frontmatter["project"],
                type=EntityType.PROJECT,
                confidence=1.0,
                source="frontmatter"
            ))

        # 4. 为标签创建实体
        for tag in result.frontmatter.get("tags", []):
            result.entities.append(Entity(
                id=hashlib.md5(f"tag:{tag}".encode()).hexdigest()[:16],
                canonical_name=tag,
                type=EntityType.CONCEPT,
                confidence=1.0,
                source="frontmatter"
            ))

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def extract_chunk_level(self, chunk_content: str, chunk_id: int,
                            note_id: str) -> ExtractionResult:
        """
        Chunk 级抽取（FR-1.3）

        使用 spacy + 规则词典，可选 LLM 后备。
        """
        start_time = time.time()
        result = ExtractionResult(source="chunk_level")

        # 1. 规则抽取（快速）
        rule_entities = self.rule_extractor.extract_entities(chunk_content)
        result.entities.extend(rule_entities)

        # 2. NLP 抽取（如果启用）
        if self.nlp_extractor:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.nlp_extractor.extract_entities, chunk_content
                    )
                    try:
                        nlp_entities = future.result(
                            timeout=self.extraction_config.chunk_timeout_ms / 1000
                        )
                        result.entities.extend(nlp_entities)
                    except FuturesTimeoutError:
                        pass
            except Exception as e:
                print(f"[DualLayerExtractor] NLP extraction error: {e}")

        # 3. 去重
        result.entities = self._deduplicate_entities(result.entities)

        # 4. LLM 后备（如果配置且召回不足）
        if (self.llm_extractor and
            len(result.entities) < self.extraction_config.llm_fallback_threshold):
            try:
                llm_entities, _ = self.llm_extractor.extract_entities(chunk_content)
                result.entities.extend(llm_entities)
            except Exception:
                pass  # 静默降级

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def create_relations_from_wikilinks(self, src_chunk_id: int,
                                        wikilinks: list[str],
                                        chunk_map: dict[str, int]) -> list[Relation]:
        """
        从 Wikilink 创建关系（FR-1.4 双层映射）

        Args:
            src_chunk_id: 源 Chunk ID
            wikilinks: Wikilink 目标列表
            chunk_map: {filepath/note_id: chunk_id} 映射

        Returns:
            关系列表
        """
        relations = []

        for link in wikilinks:
            # 查找目标 Chunk
            tgt_chunk_id = chunk_map.get(link) or chunk_map.get(f"{link}.md")

            if tgt_chunk_id:
                relation = Relation(
                    src_chunk_id=src_chunk_id,
                    tgt_chunk_id=tgt_chunk_id,
                    rel_type=RelationType.LINKS_TO,
                    scope="chunk",
                    weight=1.0,  # Wikilink 权重高
                    evidence_chunk_id=src_chunk_id,
                )
                relations.append(relation)

        return relations

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """去重实体，保留最高置信度"""
        seen = {}
        for e in entities:
            if e.id not in seen or e.confidence > seen[e.id].confidence:
                seen[e.id] = e
        return list(seen.values())


__all__ = [
    "FrontmatterParser",
    "WikilinkExtractor",
    "NLPExtractor",
    "RuleExtractor",
    "LLMExtractor",
    "DualLayerExtractor",
    "ExtractionResult",
]
