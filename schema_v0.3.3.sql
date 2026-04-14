-- 轻量级中文 RAG 系统 - 数据库 Schema (v0.3.3)
-- 新增：支持 Frontmatter 元数据存储和检索
-- 修改：FTS5 改为独立表模式，支持 jieba 分词索引
-- 修改(B7)：file_hash 移除 UNIQUE 约束，允许跨 vault 同内容文件
PRAGMA encoding = "UTF-8";
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    absolute_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size INTEGER,
    mtime INTEGER,
    is_deleted INTEGER DEFAULT 0,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER DEFAULT (strftime('%s', 'now')),
    UNIQUE(vault_name, file_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL,
    section_title TEXT,
    section_path TEXT,
    start_pos INTEGER NOT NULL,
    end_pos INTEGER NOT NULL,
    -- 以下三个字段在 v3.0+ 逻辑中可保留作为占位，但主要逻辑将转移至 confidence_json
    confidence_path_weight REAL DEFAULT 1.0,
    confidence_type_weight REAL DEFAULT 1.0,
    confidence_final_weight REAL DEFAULT 1.0, 
    
    metadata TEXT,           -- 存放完整的 Frontmatter
    confidence_json TEXT,    -- 新增：专门存放用于计算权重的原始因子 (JSON 格式)
    
    is_deleted INTEGER DEFAULT 0,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash);
CREATE INDEX IF NOT EXISTS idx_files_vault ON files(vault_name);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(content_type);
CREATE INDEX IF NOT EXISTS idx_chunks_deleted ON chunks(is_deleted);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_path);

-- FTS5 全文索引 (独立表模式)
-- rowid 与 chunks.id 关联，content 存储经 jieba 分词后的加权检索文本
-- 独立表模式支持 DELETE/INSERT，与 chunks.content 完全解耦
CREATE VIRTUAL TABLE IF NOT EXISTS fts5_index USING fts5(
    content
);

-- 向量表 (sqlite-vec)
-- 注意：此表仅在 sqlite-vec 加载成功后创建
-- 实际创建由 Python 代码动态执行

CREATE TABLE IF NOT EXISTS index_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
);

INSERT OR IGNORE INTO index_metadata (key, value) VALUES ('schema_version', '0.3.3');
INSERT OR IGNORE INTO index_metadata (key, value) VALUES ('index_status', 'valid');
INSERT OR IGNORE INTO index_metadata (key, value) VALUES ('encoding', 'UTF-8');
INSERT OR IGNORE INTO index_metadata (key, value) VALUES ('vec_support', 'true');
INSERT OR IGNORE INTO index_metadata (key, value) VALUES ('soft_delete_ratio', '0.0');
