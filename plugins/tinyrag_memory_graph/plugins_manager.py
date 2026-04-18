#!/usr/bin/env python3
"""
plugin_manager.py - 插件管理器

管理 tinyRAG 插件的生命周期，包括：
- 插件发现与注册
- 钩子调度
- 配置热加载
"""
import asyncio
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from plugins.tinyrag_memory_graph.hooks import HookContext, HookResult, HookType


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    enabled: bool = True
    instance: Any = None
    priority: int = 100


class PluginManager:
    """
    插件管理器

    负责插件的发现、注册、生命周期管理和钩子调度。

    使用示例:
        manager = PluginManager(db_conn, config_path="config.yaml")

        # 注册插件
        manager.register_plugin("memory-graph", plugin_instance)

        # 触发钩子
        result = await manager.dispatch(HookType.ON_ADD_DOCUMENT, ctx)

        # 热重载配置
        manager.reload_config()
    """

    def __init__(self, db_conn: sqlite3.Connection, config_path: str | None = None):
        """
        初始化插件管理器

        Args:
            db_conn: 数据库连接
            config_path: 配置文件路径
        """
        self.db = db_conn
        self.config_path = config_path

        # 插件注册表
        self._plugins: dict[str, PluginInfo] = {}

        # 钩子注册表：{hook_type: [(plugin_name, hook_func, priority)]}
        self._hooks: dict[HookType, list[tuple]] = field(default_factory=lambda: {
            hook: [] for hook in HookType
        })

        # 配置
        self._config: dict = {}
        if config_path:
            self._load_config()

        # 指标
        self._metrics = {
            "hooks_dispatched": 0,
            "hooks_failed": 0,
            "total_latency_ms": 0,
        }

    def _load_config(self):
        """加载配置文件"""
        if not self.config_path:
            return

        try:
            path = Path(self.config_path)
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[PluginManager] Config load error: {e}")

    def reload_config(self):
        """热重载配置"""
        self._load_config()

        # 更新所有已注册插件的配置
        for name, info in self._plugins.items():
            if info.instance and hasattr(info.instance, "config"):
                plugin_config = self._config.get("plugins", {}).get(name, {})
                if plugin_config and hasattr(info.instance.config, "from_dict"):
                    info.instance.config = type(info.instance.config).from_dict(plugin_config)

    def discover_plugins(self) -> list[str]:
        """
        发现可用插件

        通过 entry_points 和文件系统发现插件。
        """
        discovered = []

        # 1. 通过 entry_points 发现
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="tinyrag.plugins")
            for ep in eps:
                discovered.append(ep.name)
        except Exception:
            pass

        # 2. 通过文件系统发现
        plugins_dir = Path(__file__).parent.parent
        if plugins_dir.exists():
            for item in plugins_dir.iterdir():
                if item.is_dir() and not item.name.startswith("_") and (item / "__init__.py").exists():
                    discovered.append(item.name)

        return list(set(discovered))

    def register_plugin(self, name: str, plugin_instance: Any,
                        priority: int = 100) -> bool:
        """
        注册插件

        Args:
            name: 插件名称
            plugin_instance: 插件实例
            priority: 优先级（数字越小越先执行）

        Returns:
            是否注册成功
        """
        try:
            # 获取插件配置
            plugin_config = self._config.get("plugins", {}).get(name, {})
            enabled = plugin_config.get("enabled", True)

            info = PluginInfo(
                name=name,
                version=getattr(plugin_instance, "version", "unknown"),
                enabled=enabled,
                instance=plugin_instance,
                priority=priority,
            )

            self._plugins[name] = info

            # 注册钩子
            self._register_hooks(name, plugin_instance, priority)

            print(f"[PluginManager] Plugin registered: {name} (enabled={enabled})")
            return True

        except Exception as e:
            print(f"[PluginManager] Failed to register plugin {name}: {e}")
            return False

    def _register_hooks(self, plugin_name: str, plugin: Any, priority: int):
        """注册插件的所有钩子"""
        # 检查标准钩子方法
        hook_methods = {
            HookType.ON_ADD_DOCUMENT: "on_add_document",
            HookType.ON_SEARCH_AFTER: "on_search_after",
            HookType.ON_RESPONSE: "on_response",
            HookType.ON_DELETE_DOCUMENT: "on_delete_document",
            HookType.ON_REBUILD_INDEX: "on_rebuild_index",
        }

        for hook_type, method_name in hook_methods.items():
            if hasattr(plugin, method_name):
                method = getattr(plugin, method_name)
                if callable(method):
                    self._hooks[hook_type].append((plugin_name, method, priority))

    def unregister_plugin(self, name: str) -> bool:
        """注销插件"""
        if name not in self._plugins:
            return False

        # 移除钩子
        for hook_type in self._hooks:
            self._hooks[hook_type] = [
                (pn, m, p) for pn, m, p in self._hooks[hook_type]
                if pn != name
            ]

        del self._plugins[name]
        return True

    async def dispatch(self, hook_type: HookType, ctx: HookContext) -> list[HookResult]:
        """
        分发钩子事件

        按优先级顺序调用所有注册的钩子。

        Args:
            hook_type: 钩子类型
            ctx: 钩子上下文

        Returns:
            所有钩子的执行结果
        """
        results = []
        hooks = sorted(self._hooks.get(hook_type, []), key=lambda x: x[2])

        for plugin_name, hook_func, priority in hooks:
            plugin_info = self._plugins.get(plugin_name)
            if not plugin_info or not plugin_info.enabled:
                continue

            try:
                import time
                start = time.time()

                # 调用钩子
                if asyncio.iscoroutinefunction(hook_func):
                    result = await hook_func(ctx)
                else:
                    result = hook_func(ctx)

                latency = (time.time() - start) * 1000
                self._metrics["hooks_dispatched"] += 1
                self._metrics["total_latency_ms"] += latency

                results.append(result)

                # 如果钩子请求跳过后续处理
                if ctx.skip:
                    break

            except Exception as e:
                self._metrics["hooks_failed"] += 1
                results.append(HookResult.fail(f"Hook {plugin_name}.{hook_func.__name__} error: {e}"))
                print(f"[PluginManager] Hook error: {e}")

        return results

    async def initialize_plugins(self):
        """初始化所有已注册的插件"""
        for name, info in self._plugins.items():
            if info.enabled and info.instance:
                try:
                    # 设置数据库连接
                    if hasattr(info.instance, "set_db_connection"):
                        info.instance.set_db_connection(self.db)

                    # 调用初始化方法
                    if hasattr(info.instance, "initialize"):
                        await info.instance.initialize()

                except Exception as e:
                    print(f"[PluginManager] Failed to initialize plugin {name}: {e}")

    async def start_plugins(self):
        """启动所有已注册的插件"""
        for name, info in self._plugins.items():
            if info.enabled and info.instance:
                try:
                    if hasattr(info.instance, "start"):
                        await info.instance.start()
                except Exception as e:
                    print(f"[PluginManager] Failed to start plugin {name}: {e}")

    async def stop_plugins(self):
        """停止所有已注册的插件"""
        for name, info in self._plugins.items():
            if info.instance:
                try:
                    if hasattr(info.instance, "stop"):
                        await info.instance.stop()
                except Exception as e:
                    print(f"[PluginManager] Failed to stop plugin {name}: {e}")

    def get_plugin(self, name: str) -> Any | None:
        """获取插件实例"""
        info = self._plugins.get(name)
        return info.instance if info else None

    def get_plugin_info(self, name: str) -> PluginInfo | None:
        """获取插件信息"""
        return self._plugins.get(name)

    def list_plugins(self) -> list[PluginInfo]:
        """列出所有插件"""
        return list(self._plugins.values())

    def get_metrics(self) -> dict:
        """获取管理器指标"""
        return {
            **self._metrics,
            "plugins_count": len(self._plugins),
            "enabled_count": sum(1 for p in self._plugins.values() if p.enabled),
        }


# 全局插件管理器实例
_global_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器"""
    global _global_manager
    if _global_manager is None:
        raise RuntimeError("Plugin manager not initialized. Call init_plugin_manager() first.")
    return _global_manager


def init_plugin_manager(db_conn: sqlite3.Connection,
                        config_path: str | None = None) -> PluginManager:
    """初始化全局插件管理器"""
    global _global_manager
    _global_manager = PluginManager(db_conn, config_path)
    return _global_manager


__all__ = [
    "PluginInfo",
    "PluginManager",
    "get_plugin_manager",
    "init_plugin_manager",
]
