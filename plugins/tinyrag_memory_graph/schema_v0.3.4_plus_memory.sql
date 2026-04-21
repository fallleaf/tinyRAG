-- Memory Graph Plugin Schema Migration (v0.3.4)
-- 扩展 tinyRAG 核心数据库以支持图-向量混合记忆
-- 执行前提：已有 tinyRAG v0.3.3 基础表

PRAGMA encoding = "UTF-8";
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA foreign_keys = ON;

-- =====================================================
-- 1. 文档级元数据表
-- =====================================================
CREATE TABLE IF NOT EXISTS notes (
    note_id TEXT PRIMARY KEY,
    filepath TEXT UNIQUE,
    title TEXT,
    frontmatter_json TEXT,  -- JSON 格式的 Frontmatter
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_notes_filepath ON notes(filepath);
CREATE INDEX IF NOT EXISTS idx_notes_title ON notes(title);

-- =====================================================
-- 2. 扩展 chunks 表 (仅当列不存在时添加)
-- =====================================================
-- 注意：SQLite 不支持 IF NOT EXISTS for ALTER TABLE
-- 以下列应在 Python 代码中检查并添加
--
-- ALTER TABLE chunks ADD COLUMN note_id TEXT REFERENCES notes(note_id);
-- ALTER TABLE chunks ADD COLUMN inherited_meta TEXT DEFAULT '{}';
-- ALTER TABLE chunks ADD COLUMN is_representative INTEGER DEFAULT 0;
-- ALTER TABLE chunks ADD COLUMN access_count INTEGER DEFAULT 0;
-- ALTER TABLE chunks ADD COLUMN last_accessed INTEGER;

-- 创建索引 (如果列存在)
-- CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id);
-- CREATE INDEX IF NOT EXISTS idx_chunks_representative ON chunks(is_representative);

-- =====================================================
-- 3. 实体表
-- =====================================================
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    type TEXT DEFAULT 'UNKNOWN',
    confidence REAL DEFAULT 1.0,
    source TEXT DEFAULT 'nlp',
    chunk_id TEXT,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source);

-- =====================================================
-- 4. 关系表 (核心图谱结构)
-- =====================================================
CREATE TABLE IF NOT EXISTS relations (
    src_chunk_id INTEGER NOT NULL,
    tgt_chunk_id INTEGER NOT NULL,
    rel_type TEXT NOT NULL,
    scope TEXT DEFAULT 'chunk',           -- doc / chunk
    weight REAL DEFAULT 0.8,
    evidence_chunk_id INTEGER,            -- 证据来源 Chunk
    last_hit INTEGER,                     -- 最后访问时间戳
    access_count INTEGER DEFAULT 0,       -- 访问计数
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER DEFAULT (strftime('%s', 'now')),
    PRIMARY KEY (src_chunk_id, tgt_chunk_id, rel_type),
    FOREIGN KEY (src_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (tgt_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

-- 关系表索引 (性能关键)
CREATE INDEX IF NOT EXISTS idx_rel_src ON relations(src_chunk_id);
CREATE INDEX IF NOT EXISTS idx_rel_tgt ON relations(tgt_chunk_id);
CREATE INDEX IF NOT EXISTS idx_rel_weight ON relations(weight);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relations(rel_type);
CREATE INDEX IF NOT EXISTS idx_rel_scope ON relations(scope);

-- =====================================================
-- 5. 异步建图任务表
-- =====================================================
-- 注意：note_id 不使用外键约束，因为 job 可能在 note 记录创建之前就存在
-- 这是为了支持任务先于文档记录创建的场景（如批量导入时）
CREATE TABLE IF NOT EXISTS graph_build_jobs (
    job_id TEXT PRIMARY KEY,
    note_id TEXT UNIQUE,
    chunk_ids TEXT,                       -- JSON 数组
    status TEXT DEFAULT 'pending',        -- pending / processing / done / failed
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    started_at INTEGER,
    finished_at INTEGER,
    error_msg TEXT,
    retry_count INTEGER DEFAULT 0
    -- note_id 仅作为逻辑关联，不使用外键约束
    -- 原因：job 创建时 note 记录可能尚未存在，外键约束会导致插入失败
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON graph_build_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_note ON graph_build_jobs(note_id);

-- =====================================================
-- 6. 标签共现统计表 (用于原则提炼)
-- =====================================================
CREATE TABLE IF NOT EXISTS tag_co_occurrence (
    tag1 TEXT NOT NULL,
    tag2 TEXT NOT NULL,
    co_occurrence_count INTEGER DEFAULT 1,
    last_updated INTEGER DEFAULT (strftime('%s', 'now')),
    PRIMARY KEY (tag1, tag2)
);

CREATE INDEX IF NOT EXISTS idx_tag_co_count ON tag_co_occurrence(co_occurrence_count);

-- =====================================================
-- 7. 原则表 (自动提炼的知识原则)
-- =====================================================
CREATE TABLE IF NOT EXISTS principles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER NOT NULL,
    principle_text TEXT NOT NULL,
    tags TEXT,                            -- JSON 数组
    access_count INTEGER DEFAULT 0,
    is_approved INTEGER DEFAULT 0,        -- 需人工审核
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_principles_chunk ON principles(chunk_id);
CREATE INDEX IF NOT EXISTS idx_principles_approved ON principles(is_approved);

-- =====================================================
-- 8. 插件元数据（确保 index_metadata 表存在）
-- =====================================================
CREATE TABLE IF NOT EXISTS index_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
);

INSERT OR IGNORE INTO index_metadata (key, value) VALUES ('memory_graph_schema_version', '0.3.4');
INSERT OR IGNORE INTO index_metadata (key, value) VALUES ('memory_graph_enabled', 'true');
INSERT OR IGNORE INTO index_metadata (key, value) VALUES ('memory_graph_last_memify', '0');

-- =====================================================
-- 9. 视图：Chunk 完整信息 (含继承元数据)
-- =====================================================
CREATE VIEW IF NOT EXISTS v_chunks_full AS
SELECT
    c.id AS chunk_id,
    c.file_id,
    c.chunk_index,
    c.content,
    c.content_type,
    c.section_title,
    c.section_path,
    c.confidence_final_weight,
    c.metadata,
    c.confidence_json,
    f.file_path,
    f.absolute_path,
    f.vault_name,
    n.note_id,
    n.title AS note_title,
    n.frontmatter_json,
    COALESCE(c.inherited_meta, '{}') AS inherited_meta,
    c.is_representative
FROM chunks c
JOIN files f ON c.file_id = f.id
LEFT JOIN notes n ON c.note_id = n.note_id
WHERE c.is_deleted = 0 AND f.is_deleted = 0;

-- =====================================================
-- 10. 触发器：自动更新时间戳
-- =====================================================
CREATE TRIGGER IF NOT EXISTS trg_notes_updated
AFTER UPDATE ON notes
BEGIN
    UPDATE notes SET updated_at = strftime('%s', 'now') WHERE note_id = NEW.note_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_entities_updated
AFTER UPDATE ON entities
BEGIN
    UPDATE entities SET updated_at = strftime('%s', 'now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_relations_updated
AFTER UPDATE ON relations
BEGIN
    UPDATE relations SET updated_at = strftime('%s', 'now')
    WHERE src_chunk_id = NEW.src_chunk_id
      AND tgt_chunk_id = NEW.tgt_chunk_id
      AND rel_type = NEW.rel_type;
END;
