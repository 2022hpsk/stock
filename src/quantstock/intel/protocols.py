"""情报源契约与插件发现。

规范见 docs/07-信息情报模块.md 第三节、5.4。

**降级语义与行情数据刻意不同**：行情缺失必须停机（拿不准的价格算不出对的仓位），
情报缺失只降级为"缺少证据"。任一源失败只影响它覆盖的域，记 WARNING 并在日报
"情报健康"里标注，**不阻断建议生成**。

这个差别是设计上的核心判断：情报是增强项，不是前置条件。
把它做成阻断项会让系统在最需要出建议的日子（比如某个源被限流）反而沉默。
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from quantstock.infra.clock import now
from quantstock.infra.logging import get_logger
from quantstock.intel.types import IntelDomain, IntelItem, SourceHealth

__all__ = ["FetchOutcome", "NewsSource", "SourceRegistry", "discover_plugins", "fetch_all"]

_log = get_logger(__name__)


@runtime_checkable
class NewsSource(Protocol):
    """情报源。

    用户可在 ``plugins/intel_sources/`` 放置实现本协议的 Python 文件，
    启动时自动发现并注册。
    """

    # 声明成只读属性而非可变属性：可变属性在 Protocol 里是**不变**的，
    # 插件里最自然的写法 `domains = (IntelDomain.COMPANY,)` 会被推断成
    # tuple[IntelDomain]，与 tuple[IntelDomain, ...] 不兼容而报错。
    # 只读属性是协变的，普通类属性也能满足它。
    @property
    def name(self) -> str:
        """源标识。"""
        ...

    @property
    def domains(self) -> Sequence[IntelDomain]:
        """本源覆盖的情报域。"""
        ...

    def fetch(self, since: dt.datetime) -> Sequence[IntelItem]:
        """拉取增量情报。

        Args:
            since: 起始时间，tz-aware。实现方应只返回此后发布的条目。

        Returns:
            情报条目。
        """
        ...

    def health_check(self) -> SourceHealth:
        """自检。

        Returns:
            健康度。
        """
        ...


class FetchOutcome:
    """一次多源采集的结果。"""

    __slots__ = ("coverage", "items")

    def __init__(self, items: Sequence[IntelItem], coverage: dict[str, SourceHealth]) -> None:
        """初始化。

        Args:
            items: 采集到的全部条目。
            coverage: 各源健康度。
        """
        self.items = tuple(items)
        self.coverage = coverage

    @property
    def failed_sources(self) -> tuple[str, ...]:
        """失败的源名。"""
        return tuple(name for name, health in self.coverage.items() if not health.ok)

    @property
    def covered_domains(self) -> frozenset[IntelDomain]:
        """本次实际有情报的域。"""
        return frozenset(item.domain for item in self.items)

    def missing_domains(self, expected: Sequence[IntelDomain]) -> tuple[IntelDomain, ...]:
        """预期覆盖但实际没有情报的域。

        Args:
            expected: 预期覆盖的域。

        Returns:
            缺失的域。日报"情报健康"要列出来。
        """
        covered = self.covered_domains
        return tuple(d for d in expected if d not in covered)


class SourceRegistry:
    """情报源注册表。"""

    def __init__(self, sources: Sequence[NewsSource] = ()) -> None:
        """初始化。

        Args:
            sources: 初始源。
        """
        self._sources: dict[str, NewsSource] = {s.name: s for s in sources}

    def register(self, source: NewsSource) -> None:
        """注册一个源。同名覆盖。

        Args:
            source: 情报源。
        """
        self._sources[source.name] = source
        _log.info("intel_source_registered", source=source.name)

    def unregister(self, name: str) -> None:
        """注销。

        Args:
            name: 源名。
        """
        self._sources.pop(name, None)

    def all(self) -> tuple[NewsSource, ...]:
        """全部源，按名称排序以保证采集顺序可复现。

        Returns:
            源元组。
        """
        return tuple(self._sources[name] for name in sorted(self._sources))

    def for_domain(self, domain: IntelDomain) -> tuple[NewsSource, ...]:
        """覆盖指定域的源。

        Args:
            domain: 情报域。

        Returns:
            源元组。
        """
        return tuple(s for s in self.all() if domain in s.domains)

    def __len__(self) -> int:
        """已注册源数量。"""
        return len(self._sources)


def fetch_all(
    registry: SourceRegistry,
    *,
    since: dt.datetime,
    domains: Sequence[IntelDomain] | None = None,
) -> FetchOutcome:
    """并发语义的多源采集（当前为顺序实现）。

    **任一源抛异常都不会中断整体采集**——这正是情报与行情的关键差别。
    异常被收进 ``coverage`` 供日报展示。

    Args:
        registry: 源注册表。
        since: 起始时间。
        domains: 只采集覆盖这些域的源；None 表示全部。

    Returns:
        采集结果。

    Raises:
        ValueError: ``since`` 非 tz-aware。
    """
    if since.tzinfo is None:
        msg = "since 必须 tz-aware（红线 R3）"
        raise ValueError(msg)

    wanted = (
        registry.all()
        if domains is None
        else tuple(dict.fromkeys(s for d in domains for s in registry.for_domain(d)))
    )

    items: list[IntelItem] = []
    coverage: dict[str, SourceHealth] = {}
    for source in wanted:
        started = now()
        try:
            fetched = list(source.fetch(since))
        except Exception as exc:
            coverage[source.name] = SourceHealth(
                source=source.name, ok=False, error=f"{type(exc).__name__}: {exc}"
            )
            _log.warning("intel_source_failed", source=source.name, error=str(exc))
            continue

        elapsed = int((now() - started).total_seconds() * 1000)
        items.extend(fetched)
        coverage[source.name] = SourceHealth(
            source=source.name,
            ok=True,
            fetched=len(fetched),
            latency_ms=elapsed,
            last_success_at=now(),
        )

    _log.info(
        "intel_fetch_done",
        sources=len(wanted),
        items=len(items),
        failed=sum(1 for h in coverage.values() if not h.ok),
    )
    return FetchOutcome(items=items, coverage=coverage)


def discover_plugins(plugin_dir: Path) -> list[NewsSource]:
    """从目录发现自定义源插件。

    只找模块级名为 ``SOURCE`` 的实例，或以 ``Source`` 结尾且满足协议的类。
    **加载失败只警告不抛出**——一个坏插件不该让整个系统起不来。

    Args:
        plugin_dir: 插件目录，通常是 ``plugins/intel_sources``。

    Returns:
        发现的源实例。
    """
    if not plugin_dir.is_dir():
        return []

    found: list[NewsSource] = []
    for path in sorted(plugin_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            found.extend(_load_plugin(path))
        except Exception as exc:
            _log.warning("intel_plugin_load_failed", path=str(path), error=str(exc))
    return found


def _load_plugin(path: Path) -> list[NewsSource]:
    """加载单个插件文件。

    Args:
        path: 文件路径。

    Returns:
        该文件里的源实例。

    Raises:
        ImportError: 模块无法加载。
    """
    module_name = f"quantstock_intel_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"无法加载插件 {path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    out: list[NewsSource] = []
    if (instance := getattr(module, "SOURCE", None)) is not None:
        out.append(instance)
    for attr in dir(module):
        if not attr.endswith("Source") or attr.startswith("_"):
            continue
        candidate = getattr(module, attr)
        if isinstance(candidate, type) and hasattr(candidate, "name"):
            try:
                out.append(candidate())
            except TypeError:
                # 需要构造参数的类跳过：插件应导出现成的 SOURCE 实例
                _log.warning("intel_plugin_needs_args", path=str(path), cls=attr)
    return out
