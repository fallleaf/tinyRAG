#!/usr/bin/env python3
# scanner/scan_engine.py
from __future__ import annotations  # ✅ 延迟类型提示求值，彻底解决 NameError

import hashlib
import os
import time

from loguru import logger

from storage.database import DatabaseManager


class FileMeta:
    def __init__(
        self,
        vault_name: str,
        file_path: str,
        absolute_path: str,
        file_hash: str,
        file_size: int,
        mtime: int,
    ):
        self.vault_name = vault_name
        self.file_path = file_path
        self.absolute_path = absolute_path
        self.file_hash = file_hash
        self.file_size = file_size
        self.mtime = mtime

    def to_dict(self):
        return {
            "vault_name": self.vault_name,
            "file_path": self.file_path,
            "absolute_path": self.absolute_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "mtime": self.mtime,
        }


class MoveEvent:
    def __init__(self, old_path: str, new_path: str, file_hash: str, old_id: int):
        self.old_path = old_path
        self.new_path = new_path
        self.file_hash = file_hash
        self.old_id = old_id


class ScanReport:
    def __init__(self):
        self.new_files: list[FileMeta] = []
        self.modified_files: list[FileMeta] = []
        self.moved_files: list[MoveEvent] = []
        self.deleted_files: list[int] = []


class Scanner:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def calculate_hash(self, file_path: str) -> str | None:
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"❌ 计算 Hash 失败：{file_path} - {e}")
            return None

    def scan_vaults(self, vault_configs: list[tuple[str, str]]) -> ScanReport:
        report = ScanReport()
        current_scan_paths = set()

        cursor = self.db.conn.execute(
            "SELECT id, vault_name, file_path, absolute_path, file_hash, mtime, file_size FROM files WHERE is_deleted = 0"
        )
        db_files = {row["absolute_path"]: dict(row) for row in cursor.fetchall()}

        for vault_name, vault_path in vault_configs:
            vault_path = os.path.expanduser(vault_path)
            if not os.path.exists(vault_path):
                logger.warning(f"⚠️ Vault 路径不存在：{vault_path}")
                continue

            for root, _, files in os.walk(vault_path):
                for file in files:
                    if not file.endswith(".md"):
                        continue

                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, vault_path)
                    current_scan_paths.add(abs_path)

                    try:
                        stat = os.stat(abs_path)
                        mtime, size = int(stat.st_mtime), stat.st_size

                        if abs_path in db_files:
                            db_meta = db_files[abs_path]
                            if (
                                db_meta["mtime"] == mtime
                                and db_meta["file_size"] == size
                            ):
                                continue

                            new_hash = self.calculate_hash(abs_path)
                            if new_hash != db_meta["file_hash"]:
                                meta = FileMeta(
                                    vault_name,
                                    rel_path,
                                    abs_path,
                                    new_hash,
                                    size,
                                    mtime,
                                )
                                report.modified_files.append(meta)
                                logger.info(f"📝 检测到内容修改：{rel_path}")
                        else:
                            new_hash = self.calculate_hash(abs_path)
                            if not new_hash:
                                continue

                            existing = self.db.find_file_by_hash(new_hash)
                            if existing and existing["is_deleted"] == 0:
                                old_abs_path = existing["absolute_path"]
                                if old_abs_path in current_scan_paths:
                                    meta = FileMeta(
                                        vault_name,
                                        rel_path,
                                        abs_path,
                                        new_hash,
                                        size,
                                        mtime,
                                    )
                                    report.new_files.append(meta)
                                    logger.info(f"➕ 检测到复制文件: {rel_path}")
                                else:
                                    report.moved_files.append(
                                        MoveEvent(
                                            old_path=existing["file_path"],
                                            new_path=rel_path,
                                            file_hash=new_hash,
                                            old_id=existing["id"],
                                        )
                                    )
                                    logger.info(
                                        f"🔄 检测到文件移动：{existing['file_path']} -> {rel_path}"
                                    )
                            else:
                                meta = FileMeta(
                                    vault_name,
                                    rel_path,
                                    abs_path,
                                    new_hash,
                                    size,
                                    mtime,
                                )
                                report.new_files.append(meta)
                                logger.info(f"➕ 检测到新文件：{rel_path}")

                    except Exception as e:
                        logger.error(f"❌ 处理文件失败：{abs_path} - {e}")

        for abs_path, db_meta in db_files.items():
            if abs_path not in current_scan_paths:
                report.deleted_files.append(db_meta["id"])
                logger.info(f"🗑️ 检测到文件删除：{db_meta['file_path']}")

        return report

    def process_report(self, report: ScanReport):
        try:
            for meta in report.new_files:
                existing = self.db.find_file_by_hash(
                    meta.file_hash, include_deleted=True
                )
                if existing and existing["is_deleted"] == 1:
                    self.db.conn.execute(
                        """UPDATE files SET
                           vault_name=?, file_path=?, absolute_path=?, file_size=?, mtime=?, is_deleted=0, updated_at=?
                           WHERE id=?""",
                        (
                            meta.vault_name,
                            meta.file_path,
                            meta.absolute_path,
                            meta.file_size,
                            meta.mtime,
                            int(time.time()),
                            existing["id"],
                        ),
                    )
                elif existing and existing["is_deleted"] == 0:
                    logger.debug(f"⚠️ 跳过重复 Hash 文件：{meta.file_path}")
                else:
                    self.db.upsert_file(meta.to_dict())

            for meta in report.modified_files:
                cursor = self.db.conn.execute(
                    "SELECT id FROM files WHERE absolute_path = ?",
                    (meta.absolute_path,),
                )
                row = cursor.fetchone()
                if row:
                    self.db.conn.execute(
                        "UPDATE chunks SET is_deleted = 1 WHERE file_id = ?",
                        (row["id"],),
                    )
                    self.db.upsert_file(meta.to_dict())

            for move in report.moved_files:
                self.db.conn.execute(
                    "UPDATE files SET file_path = ?, updated_at = ? WHERE id = ?",
                    (move.new_path, int(time.time()), move.old_id),
                )

            if report.deleted_files:
                # ✅ 修复：参数化 IN 查询，避免 SQL 注入与语法错误
                placeholders = ", ".join(["?"] * len(report.deleted_files))
                self.db.conn.execute(
                    f"UPDATE files SET is_deleted = 1 WHERE id IN ({placeholders})",
                    report.deleted_files,
                )
                self.db.conn.execute(
                    f"UPDATE chunks SET is_deleted = 1 WHERE file_id IN ({placeholders})",
                    report.deleted_files,
                )

            self.db.conn.commit()
            logger.success("✅ 扫描报告处理完成")
        except Exception as e:
            self.db.conn.rollback()
            logger.error(f"❌ 数据库更新失败：{e}")


__all__ = ["FileMeta", "MoveEvent", "ScanReport", "Scanner"]
