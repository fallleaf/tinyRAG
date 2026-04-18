#!/usr/bin/env python3
"""
plugins/bootstrap.py - 插件引导加载器 (v1.0)

功能:
- 自动发现 plugins/ 目录下的插件包
- 根据配置动态加载启用的插件
- 提供 PluginBase 基类供插件继承
- 零侵入核心：通过钩子协议扩展功能

使用方式:
    from plugins.bootstrap import PluginLoader, PluginBase

    loader = PluginLoader(config)
    loader.load_all()

    # 触发钩子
    loader.hook('on_file_indexed', file_id=123, chunks=[...])
"""
import importlib
import importlib.util
import json
import sys
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import logger

# 插件目录
PLUGINS_DIR = Path(__file__).parent


@dataclass
class PluginInfo:
    """插件元信息"""
    name: str
    version: str
    description: str = ""
    author: str = ""
    enabled: bool = True
    priority: int = 100  # 加载优先级，数值越小越先加载
    dependencies: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "PluginInfo":
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 100),
            dependencies=data.get("dependencies", []),
        )


class PluginBase(ABC):
    """
    插件基类 - 所有插件必须继承此类

    生命周期:
    1. __init__(config, context) - 初始化
    2. on_load() - 加载时调用
    3. on_enable() - 启用时调用
    4. on_disable() - 禁用时调用
    5. on_unload() - 卸载时调用
    """

    # 插件元信息 (子类必须覆盖)
    NAME: str = "base_plugin"
    VERSION: str = "0.0.0"
    DESCRIPTION: str = "Base plugin class"

    def __init__(self, config: Any = None, context: Any = None):
        """
        初始化插件

        Args:
            config: 全局 Settings 配置对象
            context: AppContext 上下文对象 (包含 db, retriever 等)
        """
        self.config = config
        self.ctx = context
        self._enabled = False
        self._hooks: dict[str, list[Callable]] = {}

    @property
    def info(self) -> PluginInfo:
        """获取插件信息"""
        return PluginInfo(
            name=self.NAME,
            version=self.VERSION,
            description=self.DESCRIPTION,
        )

    def on_load(self) -> bool:
        """
        插件加载时调用 (只执行一次)

        Returns:
            bool: 加载成功返回 True
        """
        return True

    def on_enable(self) -> bool:
        """
        插件启用时调用

        Returns:
            bool: 启用成功返回 True
        """
        return True

    def on_disable(self) -> None:
        """插件禁用时调用"""
        pass

    def on_unload(self) -> None:
        """插件卸载时调用"""
        pass

    def register_hook(self, event: str, handler: Callable) -> None:
        """注册事件钩子"""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(handler)

    def get_hooks(self, event: str) -> list[Callable]:
        """获取指定事件的所有钩子"""
        return self._hooks.get(event, [])

    def register_tool(self, tool_cls: type) -> None:
        """
        注册 MCP 工具 (由 PluginLoader 调用)

        Args:
            tool_cls: 工具类，需继承 BaseTool
        """
        # 由 PluginLoader 在加载时处理
        pass


class PluginLoader:
    """
    插件加载器 - 负责发现、加载、管理插件

    使用示例:
        loader = PluginLoader(settings)
        loader.load_all()

        # 触发钩子
        loader.invoke_hook('on_file_indexed', file_id=1, chunks=[...])

        # 获取插件实例
        plugin = loader.get_plugin('tinyrag_memory_graph')
    """

    def __init__(self, config: Any, context: Any = None):
        """
        初始化加载器

        Args:
            config: 全局 Settings 配置
            context: AppContext 上下文
        """
        self.config = config
        self.ctx = context
        self._plugins: dict[str, PluginBase] = {}
        self._plugin_configs: dict[str, dict] = {}
        self._load_plugin_configs()

    def _load_plugin_configs(self) -> None:
        """从配置中加载插件配置项"""
        # 从全局配置中获取 plugins 配置
        if hasattr(self.config, 'plugins'):
            for plugin_cfg in self.config.plugins:
                name = plugin_cfg.get('name') if isinstance(plugin_cfg, dict) else getattr(plugin_cfg, 'name', None)
                if name:
                    self._plugin_configs[name] = plugin_cfg if isinstance(plugin_cfg, dict) else plugin_cfg.__dict__

    def discover_plugins(self) -> list[PluginInfo]:
        """
        发现 plugins/ 目录下的所有插件包

        Returns:
            list[PluginInfo]: 发现的插件列表
        """
        discovered = []

        if not PLUGINS_DIR.exists():
            logger.warning(f"插件目录不存在: {PLUGINS_DIR}")
            return discovered

        for plugin_path in PLUGINS_DIR.iterdir():
            if not plugin_path.is_dir():
                continue
            if plugin_path.name.startswith('_') or plugin_path.name.startswith('.'):
                continue
            if plugin_path.name == '__pycache__':
                continue

            # 检查是否有 __init__.py 或 plugin.py
            init_file = plugin_path / "__init__.py"
            plugin_file = plugin_path / "plugin.py"

            if not (init_file.exists() or plugin_file.exists()):
                continue

            # 尝试读取插件元信息
            info = self._read_plugin_info(plugin_path)
            if info:
                discovered.append(info)

        logger.info(f"🔍 发现 {len(discovered)} 个插件: {[p.name for p in discovered]}")
        return discovered

    def _read_plugin_info(self, plugin_path: Path) -> PluginInfo | None:
        """读取插件元信息"""
        # 尝试从 plugin.json 读取
        meta_file = plugin_path / "plugin.json"
        if meta_file.exists():
            try:
                with open(meta_file, encoding='utf-8') as f:
                    meta = json.load(f)
                return PluginInfo.from_dict(meta)
            except Exception as e:
                logger.warning(f"读取插件元信息失败 {meta_file}: {e}")

        # 尝试从 __init__.py 或 plugin.py 中读取常量
        for filename in ["plugin.py", "__init__.py"]:
            file_path = plugin_path / filename
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    # 简单解析常量
                    name = self._extract_constant(content, "NAME", plugin_path.name)
                    version = self._extract_constant(content, "VERSION", "0.0.0")
                    desc = self._extract_constant(content, "DESCRIPTION", "")
                    return PluginInfo(name=name, version=version, description=desc)
                except Exception as e:
                    logger.warning(f"解析插件文件失败 {file_path}: {e}")

        # 使用目录名作为插件名
        return PluginInfo(name=plugin_path.name, version="0.0.0")

    def _extract_constant(self, content: str, const_name: str, default: str) -> str:
        """从 Python 文件中提取常量值"""
        import re
        pattern = rf'^{const_name}\s*=\s*["\']([^"\']+)["\']'
        match = re.search(pattern, content, re.MULTILINE)
        return match.group(1) if match else default

    def load_plugin(self, plugin_name: str) -> PluginBase | None:
        """
        加载单个插件

        Args:
            plugin_name: 插件名称

        Returns:
            PluginBase | None: 加载成功返回插件实例
        """
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]

        plugin_path = PLUGINS_DIR / plugin_name
        if not plugin_path.exists():
            logger.error(f"插件不存在: {plugin_name}")
            return None

        try:
            # 动态导入插件模块
            module_name = f"plugins.{plugin_name}"

            # 方法：将插件目录添加到 sys.path，然后使用 import_module
            # 这是最可靠的方式，能正确处理包内的所有导入
            plugins_parent = str(PLUGINS_DIR.parent)
            if plugins_parent not in sys.path:
                sys.path.insert(0, plugins_parent)

            # 直接导入插件包（这会加载 __init__.py 并注册包到 sys.modules）
            try:
                plugin_pkg = importlib.import_module(module_name)
            except ModuleNotFoundError:
                # 如果包没有 __init__.py，创建一个占位模块
                import types
                plugin_pkg = types.ModuleType(module_name)
                plugin_pkg.__path__ = [str(plugin_path)]
                plugin_pkg.__package__ = module_name
                sys.modules[module_name] = plugin_pkg

            # 导入 plugin.py 模块
            plugin_module_path = plugin_path / "plugin.py"
            module = importlib.import_module(f"{module_name}.plugin") if plugin_module_path.exists() else plugin_pkg

            # 查找 PluginBase 子类
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, PluginBase) and
                    attr is not PluginBase):
                    plugin_class = attr
                    break

            if not plugin_class:
                logger.error(f"插件 {plugin_name} 未找到 PluginBase 子类")
                return None

            # 获取插件配置
            plugin_cfg = self._plugin_configs.get(plugin_name, {})

            # 实例化插件
            plugin_instance = plugin_class(
                config=plugin_cfg.get('config', {}),
                context=self.ctx
            )

            # 调用 on_load
            if not plugin_instance.on_load():
                logger.error(f"插件 {plugin_name} on_load() 返回 False")
                return None

            self._plugins[plugin_name] = plugin_instance
            logger.info(f"✅ 插件加载成功: {plugin_name} v{plugin_instance.VERSION}")

            return plugin_instance

        except Exception as e:
            logger.error(f"加载插件 {plugin_name} 失败: {e}", exc_info=True)
            return None

    def load_all(self) -> dict[str, PluginBase]:
        """
        加载所有启用的插件

        Returns:
            dict[str, PluginBase]: 加载成功的插件字典
        """
        discovered = self.discover_plugins()

        # 按优先级排序
        discovered.sort(key=lambda p: p.priority)

        for info in discovered:
            # 检查是否启用
            plugin_cfg = self._plugin_configs.get(info.name, {})
            enabled = plugin_cfg.get('enabled', info.enabled)

            # 如果配置中有显式设置，以配置为准
            if 'enabled' in plugin_cfg:
                enabled = plugin_cfg['enabled']

            if not enabled:
                logger.info(f"⏭️ 插件 {info.name} 已禁用，跳过加载")
                continue

            # 检查依赖
            missing_deps = []
            for dep in info.dependencies:
                if dep not in self._plugins:
                    missing_deps.append(dep)

            if missing_deps:
                logger.warning(f"⚠️ 插件 {info.name} 缺少依赖: {missing_deps}，跳过加载")
                continue

            self.load_plugin(info.name)

        # 启用所有已加载的插件
        for name, plugin in self._plugins.items():
            if plugin.on_enable():
                plugin._enabled = True

        logger.info(f"🎉 共加载 {len(self._plugins)} 个插件")
        return self._plugins

    def get_plugin(self, name: str) -> PluginBase | None:
        """获取已加载的插件实例"""
        return self._plugins.get(name)

    def get_all_plugins(self) -> dict[str, PluginBase]:
        """获取所有已加载的插件"""
        return self._plugins

    def invoke_hook(self, event: str, **kwargs) -> list[Any]:
        """
        触发钩子事件，调用所有插件的对应钩子

        Args:
            event: 事件名称
            **kwargs: 传递给钩子的参数

        Returns:
            list[Any]: 所有钩子的返回值列表
        """
        import asyncio
        import inspect

        results = []
        pending_tasks = []  # 收集需要等待的异步任务

        for name, plugin in self._plugins.items():
            if not plugin._enabled:
                continue

            hooks = plugin.get_hooks(event)
            for hook in hooks:
                try:
                    result = hook(**kwargs)
                    # 如果是异步函数，需要用 asyncio 运行
                    if inspect.iscoroutine(result):
                        try:
                            # 尝试获取当前事件循环
                            loop = asyncio.get_running_loop()
                            # 如果已在异步上下文中，创建任务并保存引用
                            task = asyncio.create_task(result)
                            pending_tasks.append((name, task))
                        except RuntimeError:
                            # 没有运行中的事件循环，创建新的
                            result = asyncio.run(result)
                            results.append(result)
                    else:
                        results.append(result)
                except Exception as e:
                    logger.error(f"插件 {name} 钩子 {event} 执行失败: {e}")

        # 如果有挂起的异步任务，需要等待它们完成
        # 注意：这只能在同步上下文中通过创建新事件循环来处理
        if pending_tasks:
            try:
                # 检查是否已在异步上下文中
                loop = asyncio.get_running_loop()
                # 如果在异步上下文中，任务已提交，等待完成
                async def wait_tasks():
                    for name, task in pending_tasks:
                        try:
                            await task
                        except Exception as e:
                            logger.error(f"插件 {name} 异步钩子 {event} 执行失败: {e}")
                # 创建一个包装任务来等待所有任务
                _background_task = asyncio.create_task(wait_tasks())  # noqa: RUF006
                # 注意：在同步上下文中我们无法等待异步任务完成
                # 这种情况下任务会在后台执行
            except RuntimeError:
                # 没有运行中的事件循环，创建新的来执行所有任务
                async def run_all_tasks():
                    task_results = []
                    for name, task in pending_tasks:
                        try:
                            r = await task
                            task_results.append(r)
                        except Exception as e:
                            logger.error(f"插件 {name} 异步钩子 {event} 执行失败: {e}")
                    return task_results
                results.extend(asyncio.run(run_all_tasks()))

        return results

    def register_tools_to_registry(self, registry: Any) -> int:
        """
        将插件工具注册到 MCP ToolRegistry

        Args:
            registry: ToolRegistry 实例

        Returns:
            int: 注册的工具数量
        """
        count = 0
        for name, plugin in self._plugins.items():
            if not plugin._enabled:
                continue

            # 检查插件是否定义了 TOOLS 属性
            if hasattr(plugin, 'TOOLS'):
                for tool_cls in plugin.TOOLS:
                    try:
                        registry.register(tool_cls)
                        count += 1
                        logger.info(f"✅ 注册工具: {tool_cls.name if hasattr(tool_cls, 'name') else tool_cls.__name__}")
                    except Exception as e:
                        logger.error(f"注册工具失败: {e}")

        return count

    def shutdown(self) -> None:
        """关闭所有插件"""
        for name, plugin in self._plugins.items():
            try:
                plugin.on_disable()
                plugin.on_unload()
            except Exception as e:
                logger.error(f"关闭插件 {name} 失败: {e}")

        self._plugins.clear()
        logger.info("所有插件已关闭")


# =====================
# 便捷函数
# =====================
_global_loader: PluginLoader | None = None


def get_plugin_loader() -> PluginLoader | None:
    """获取全局插件加载器"""
    return _global_loader


def init_plugins(config: Any, context: Any = None) -> PluginLoader:
    """
    初始化插件系统

    Args:
        config: 全局配置
        context: 应用上下文

    Returns:
        PluginLoader: 插件加载器实例
    """
    global _global_loader
    _global_loader = PluginLoader(config, context)
    _global_loader.load_all()
    return _global_loader


def shutdown_plugins() -> None:
    """关闭插件系统"""
    global _global_loader
    if _global_loader:
        _global_loader.shutdown()
        _global_loader = None
