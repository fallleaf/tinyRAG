#!/usr/bin/env python3
# scanner/scan_engine.py - 文件扫描引擎 (v2.0)
"""
v2.0 优化内容:
1. ✅ 两阶段扫描：先收集路径 + stat，再按需计算 Hash，避免无效 I/O
2. ✅ 可靠的移动检测：基于「消失文件集」匹配，不再依赖 os.walk 遍历顺序
3. ✅ MoveEvent 增加完整元数据（vault_name, absolute_path, mtime, size）
4. ✅ 移动处理同步更新 absolute_path / vault_name / mtime / file_size
5. ✅ 删除处理同步清理 FTS5 / vectors 关联数据
6. ✅ 修改处理同步清理 FTS5 / vectors 关联数据
7. ✅ 跳过隐藏目录（.git, .obsidian 等），减少无效扫描
8. ✅ Hash 读缓冲区从 8KB 提升到 64KB
9. ✅ DB 查询按 vault_name 过滤，减少内存占用
10. ✅ 使用 dataclass 简化 FileMeta / MoveEvent 定义
11. ✅ 新增 touched_files 机制，避免 mtime-only 变化触发重复 Hash
12. ✅ 空报告快速返回
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from storage.database import DatabaseManager

# 默认跳过的目录名（不进入子目录扫描）
DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".obsidian",
        ".trash",
        ".Trash",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    }
)


@dataclass
class FileMeta:
    """文件元数据"""

    vault_name: str
    file_path: str
    absolute_path: str
    file_hash: str
    file_size: int
    mtime: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "vault_name": self.vault_name,
            "file_path": self.file_path,
            "absolute_path": self.absolute_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "mtime": self.mtime,
        }


@dataclass
class MoveEvent:
    """文件移动事件（含完整的新旧位置元数据）"""

    old_id: int
    old_path: str
    old_vault_name: str
    new_path: str
    new_vault_name: str
    new_absolute_path: str
    file_hash: str
    new_mtime: int
    new_file_size: int


class ScanReport:
    """扫描报告：汇总新增、修改、移动、删除的文件"""

    def __init__(self):
        self.new_files: list[FileMeta] = []
        self.modified_files: list[FileMeta] = []
        self.moved_files: list[MoveEvent] = []
        self.deleted_files: list[int] = []
        # 仅 mtime/size 变化但内容未变的文件：(db_id, mtime, file_size)
        self.touched_files: list[tuple[int, int, int]] = []

    def summary(self) -> str:
        return (
            f"新增: {len(self.new_files)}, "
            f"修改: {len(self.modified_files)}, "
            f"移动: {len(self.moved_files)}, "
            f"删除: {len(self.deleted_files)}, "
            f"仅时间戳更新: {len(self.touched_files)}"
        )


class Scanner:
    """文件扫描引擎：检测 vault 目录中的文件变更"""

    def __init__(
        self,
        db: DatabaseManager,
        skip_dirs: frozenset[str] | None = None,
    ):
        self.db = db
        self._skip_dirs = skip_dirs or DEFAULT_SKIP_DIRS

    @staticmethod
    def calculate_hash(file_path: str) -> str | None:
        """计算文件的 SHA-256 哈希值（64KB 缓冲区）"""
        try:
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    sha256.update(block)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"❌ 计算哈希失败：{file_path} - {e}")
            return None

    def _walk_vaults(self, vault_configs: list[tuple[str, str]]) -> dict[str, tuple[str, str, int, int]]:
        """
        阶段 1 — 轻量路径收集：
        遍历所有 vault，收集磁盘上 .md 文件的路径和 stat 信息。
        跳过隐藏目录，不计算 hash，纯 I/O 遍历。

        返回: {absolute_path: (vault_name, rel_path, mtime, file_size)}
        """
        disk_files: dict[str, tuple[str, str, int, int]] = {}

        for vault_name, vault_path in vault_configs:
            vault_path = os.path.expanduser(vault_path)
            if not os.path.isdir(vault_path):
                logger.warning(f"⚠️ Vault 路径不存在：{vault_path}")
                continue

            for root, dirs, files in os.walk(vault_path):
                # 就地过滤：阻止 os.walk 递归进入隐藏/无关目录
                dirs[:] = sorted(d for d in dirs if d not in self._skip_dirs)
                for fname in sorted(files):
                    if not fname.endswith(".md"):
                        continue

                    abs_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_path, vault_path)

                    try:
                        stat = os.stat(abs_path)
                        disk_files[abs_path] = (
                            vault_name,
                            rel_path,
                            int(stat.st_mtime),
                            stat.st_size,
                        )
                    except OSError as e:
                        logger.warning(f"⚠️ 无法读取文件状态：{abs_path} - {e}")

        return disk_files

    def scan_vaults(self, vault_configs: list[tuple[str, str]]) -> ScanReport:
        """
        两阶段扫描：
        阶段 1 — 轻量遍历：收集磁盘文件路径 + stat（无 hash I/O）
        阶段 2 — 差异检测：仅对变化文件计算 hash，与 DB 对比分类

        移动检测策略：
        - 构建「消失文件集」= DB 中存在但磁盘上不存在的文件
        - 对每个新增文件，检查其 hash 是否匹配消失文件
        - 匹配成功 → 移动事件；匹配失败 → 新增文件
        - 此策略不依赖遍历顺序，结果确定且正确
        """
        report = ScanReport()

        if not vault_configs:
            return report

        # 加载 DB 中属于当前扫描 vault 的所有未删除文件（按 vault 过滤减少内存）
        scanned_vaults = [v[0] for v in vault_configs]
        placeholders = ", ".join(["?"] * len(scanned_vaults))
        sql = (
            "SELECT id, vault_name, file_path, absolute_path, "
            "file_hash, mtime, file_size "
            f"FROM files WHERE is_deleted = 0 AND vault_name IN ({placeholders})"
        )
        cursor = self.db.conn.execute(sql, scanned_vaults)
        db_files: dict[str, dict[str, Any]] = {row["absolute_path"]: dict(row) for row in cursor.fetchall()}

        # ═══ 阶段 1：轻量路径收集 ═══
        disk_files = self._walk_vaults(vault_configs)
        disk_paths = set(disk_files.keys())
        db_paths = set(db_files.keys())

        # 消失文件集：DB 中有但磁盘上没有（可能被删除或被移动到其他位置）
        disappeared: dict[str, dict[str, Any]] = {ap: meta for ap, meta in db_files.items() if ap not in disk_paths}

        # hash → 消失文件的反向索引（每个 hash 只保留首个匹配，避免歧义）
        disappeared_by_hash: dict[str, dict[str, Any]] = {}
        for meta in disappeared.values():
            h = meta["file_hash"]
            if h not in disappeared_by_hash:
                disappeared_by_hash[h] = meta

        # ═══ 阶段 2a：修改检测 — 路径相同，mtime/size 变化 ═══
        for abs_path in disk_paths & db_paths:
            db_meta = db_files[abs_path]
            vault_name, rel_path, mtime, size = disk_files[abs_path]

            # mtime 和 size 都未变 → 文件未修改，跳过
            if db_meta["mtime"] == mtime and db_meta["file_size"] == size:
                continue

            new_hash = self.calculate_hash(abs_path)
            if new_hash is None:
                continue

            if new_hash != db_meta["file_hash"]:
                # 内容确实变化 → 标记为修改
                report.modified_files.append(FileMeta(vault_name, rel_path, abs_path, new_hash, size, mtime))
                logger.info(f"📝 检测到内容修改：{rel_path}")
            else:
                # mtime/size 变化但内容未变 → 仅更新时间戳
                report.touched_files.append((db_meta["id"], mtime, size))
                logger.debug(f"⏱️ 仅时间戳变化：{rel_path}")

        # ═══ 阶段 2b：新增 / 移动检测 — 磁盘上有但 DB 中没有 ═══
        for abs_path in disk_paths - db_paths:
            vault_name, rel_path, mtime, size = disk_files[abs_path]
            new_hash = self.calculate_hash(abs_path)
            if new_hash is None:
                continue

            # 检查是否匹配某个消失文件（移动检测）
            src = disappeared_by_hash.get(new_hash)
            if src and src["absolute_path"] in disappeared:
                report.moved_files.append(
                    MoveEvent(
                        old_id=src["id"],
                        old_path=src["file_path"],
                        old_vault_name=src["vault_name"],
                        new_path=rel_path,
                        new_vault_name=vault_name,
                        new_absolute_path=abs_path,
                        file_hash=new_hash,
                        new_mtime=mtime,
                        new_file_size=size,
                    )
                )
                logger.info(
                    f"🔄 检测到文件移动：" f"{src['vault_name']}/{src['file_path']} → " f"{vault_name}/{rel_path}"
                )
                # 从消失集中移除，避免同一源文件被重复匹配
                disappeared.pop(src["absolute_path"], None)
            else:
                report.new_files.append(FileMeta(vault_name, rel_path, abs_path, new_hash, size, mtime))
                logger.info(f"➕ 检测到新文件：{rel_path}")

        # ═══ 阶段 2c：删除检测 — 未被匹配为移动源的消失文件 ═══
        for abs_path, meta in disappeared.items():
            report.deleted_files.append(meta["id"])
            logger.info(f"🗑️ 检测到文件删除：{meta['vault_name']}/{meta['file_path']}")

        logger.info(f"📊 扫描结果：{report.summary()}")
        return report

    def process_report(self, report: ScanReport) -> None:
        """
        将 ScanReport 持久化到数据库。
        包含 FTS5 / vectors 联动清理，确保删除/修改操作不残留孤立数据。
        """
        total = (
            len(report.new_files)
            + len(report.modified_files)
            + len(report.moved_files)
            + len(report.deleted_files)
            + len(report.touched_files)
        )
        if total == 0:
            logger.info("✨ 扫描报告为空，无需更新")
            return

        try:
            # ── 新增文件 ──────────────────────────────────
            for meta in report.new_files:
                existing = self.db.find_file_by_hash(meta.file_hash, include_deleted=True)
                if existing and existing["is_deleted"] == 1:
                    # 恢复已软删除的记录（避免重复 INSERT + 保留历史关联）
                    self.db.conn.execute(
                        """UPDATE files SET
                           vault_name=?, file_path=?, absolute_path=?,
                           file_hash=?, file_size=?, mtime=?,
                           is_deleted=0, updated_at=?
                           WHERE id=?""",
                        (
                            meta.vault_name,
                            meta.file_path,
                            meta.absolute_path,
                            meta.file_hash,
                            meta.file_size,
                            meta.mtime,
                            int(time.time()),
                            existing["id"],
                        ),
                    )
                    logger.debug(f"♻️ 恢复软删除记录：{meta.file_path}")
                elif existing and existing["is_deleted"] == 0:
                    # Hash 已存在且未删除 → 可能是复制文件，跳过避免重复
                    logger.debug(f"⚠️ 跳过重复 Hash 文件：{meta.file_path}")
                else:
                    self.db.upsert_file(meta.to_dict())

            # ── 修改文件（含 FTS5/vectors 联动清理）─────────
            for meta in report.modified_files:
                cursor = self.db.conn.execute(
                    "SELECT id FROM files WHERE absolute_path = ?",
                    (meta.absolute_path,),
                )
                row = cursor.fetchone()
                if row:
                    file_id = row["id"]
                    self._soft_delete_chunks(file_id)
                    self.db.upsert_file(meta.to_dict())
                    logger.debug(f"📝 已更新文件元数据：{meta.file_path}")

            # ── 移动文件（完整更新所有路径相关字段）─────────
            for move in report.moved_files:
                self.db.conn.execute(
                    """UPDATE files SET
                       file_path=?, absolute_path=?, vault_name=?,
                       file_size=?, mtime=?, updated_at=?
                       WHERE id=?""",
                    (
                        move.new_path,
                        move.new_absolute_path,
                        move.new_vault_name,
                        move.new_file_size,
                        move.new_mtime,
                        int(time.time()),
                        move.old_id,
                    ),
                )
                logger.debug(
                    f"🔄 已更新移动文件："
                    f"{move.old_vault_name}/{move.old_path} → "
                    f"{move.new_vault_name}/{move.new_path}"
                )

            # ── 删除文件（含 FTS5/vectors 联动清理）─────────
            if report.deleted_files:
                ph = ", ".join(["?"] * len(report.deleted_files))
                now = int(time.time())

                # 软删除 files（同时更新 updated_at）
                self.db.conn.execute(
                    f"UPDATE files SET is_deleted=1, updated_at=? WHERE id IN ({ph})",
                    [now, *report.deleted_files],
                )
                # 软删除关联 chunks
                self.db.conn.execute(
                    f"UPDATE chunks SET is_deleted=1 WHERE file_id IN ({ph})",
                    report.deleted_files,
                )
                # 清理 FTS5 独立表（释放索引空间）
                try:
                    self.db.conn.execute(
                        f"DELETE FROM fts5_index WHERE rowid IN (" f"SELECT id FROM chunks WHERE file_id IN ({ph}))",
                        report.deleted_files,
                    )
                except Exception as e:
                    logger.warning(f"⚠️ FTS5 清理失败（可忽略）：{e}")
                # 清理 vectors（释放向量存储空间）
                try:
                    self.db.conn.execute(
                        f"DELETE FROM vectors WHERE chunk_id IN (" f"SELECT id FROM chunks WHERE file_id IN ({ph}))",
                        report.deleted_files,
                    )
                except Exception as e:
                    logger.warning(f"⚠️ vectors 清理失败（可忽略）：{e}")

            # ── 仅时间戳变化的文件 ─────────────────────────
            if report.touched_files:
                self.db.conn.executemany(
                    "UPDATE files SET mtime=?, file_size=? WHERE id=?",
                    report.touched_files,
                )

            self.db.conn.commit()
            logger.success(f"✅ 扫描报告处理完成（共 {total} 项变更）")
        except Exception as e:
            self.db.conn.rollback()
            logger.error(f"❌ 数据库更新失败：{e}", exc_info=True)

    def _soft_delete_chunks(self, file_id: int) -> None:
        """软删除文件关联的 chunks，并同步清理 FTS5 / vectors"""
        self.db.conn.execute(
            "UPDATE chunks SET is_deleted=1 WHERE file_id=?",
            (file_id,),
        )
        # 清理 FTS5 独立表
        try:
            self.db.conn.execute(
                "DELETE FROM fts5_index WHERE rowid IN (" "SELECT id FROM chunks WHERE file_id=? AND is_deleted=1)",
                (file_id,),
            )
        except Exception as e:
            logger.warning(f"⚠️ FTS5 清理失败（可忽略）：{e}")
        # 清理 vectors
        try:
            self.db.conn.execute(
                "DELETE FROM vectors WHERE chunk_id IN (" "SELECT id FROM chunks WHERE file_id=? AND is_deleted=1)",
                (file_id,),
            )
        except Exception as e:
            logger.warning(f"⚠️ vectors 清理失败（可忽略）：{e}")


__all__ = ["FileMeta", "MoveEvent", "ScanReport", "Scanner"]
