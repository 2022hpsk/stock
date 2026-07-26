"""情报服务：采集、导入、摘要、黑名单查询。

CLI 与界面共用同一实现。

**关键行为**：任何一步失败都不阻断——情报是增强项。采集失败记进 coverage，
日报里标注"今日 XX 域情报缺失"，建议照常生成（docs/07 第三节）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantstock.config.settings import Settings
from quantstock.infra.clock import now, today
from quantstock.infra.logging import get_logger
from quantstock.infra.types import Symbol, TradeDate
from quantstock.intel.blacklist import BlacklistEntry, IntelBlacklist
from quantstock.intel.digest import DigestBuilder
from quantstock.intel.entity import EntityLinker, SymbolDictionary
from quantstock.intel.external import (
    ImportReport,
    InboxScanner,
    build_item,
    items_from_rows,
    parse_payload,
)
from quantstock.intel.pipeline import IntelPipeline, PipelineResult
from quantstock.intel.protocols import NewsSource, SourceRegistry, discover_plugins, fetch_all
from quantstock.intel.scoring import ImportanceScorer
from quantstock.intel.sources import ClsTelegraphSource, EastMoneySource, RssSource
from quantstock.intel.sources.rss import RssFeed
from quantstock.intel.store import IntelStore
from quantstock.intel.types import IntelDigest, IntelDomain, IntelItem, SourceHealth

# CLI 与界面只允许依赖 services（F20.1 分层契约），情报契约在这里转出。
__all__ = [
    "BlacklistEntry",
    "IntelDigest",
    "IntelDomain",
    "IntelItem",
    "IntelService",
    "IntelStatus",
    "PipelineResult",
    "parse_payload",
]

_log = get_logger(__name__)

DEFAULT_LOOKBACK_DAYS = 7

MAX_TRACKED_SYMBOLS = 50
"""逐只查询个股新闻的标的上限。

东方财富的个股新闻接口按标的逐个请求，礼貌抓取限速下（默认 30 次/分）
一百只标的就要跑三分多钟。持仓加候选池超过这个数时截断并记警告——
让采集慢到超时，比少查几只标的的新闻更糟。
"""


@dataclass(frozen=True, slots=True)
class IntelStatus:
    """情报模块状态，供仪表盘与 ``quantstock status``。"""

    sources: int
    inbox_pending: int
    latest_date: TradeDate | None
    blacklisted: int
    dates_available: int

    @property
    def message(self) -> str:
        """一行摘要。"""
        latest = self.latest_date.isoformat() if self.latest_date else "无"
        return (
            f"情报源 {self.sources} 个，收件箱待处理 {self.inbox_pending} 个，"
            f"最新情报日 {latest}，黑名单 {self.blacklisted} 只"
        )


class IntelService:
    """情报服务。"""

    def __init__(
        self,
        settings: Settings,
        *,
        registry: SourceRegistry | None = None,
        dictionaries: Iterable[SymbolDictionary] = (),
        holdings: Iterable[Symbol] = (),
        watchlist: Iterable[Symbol] = (),
    ) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
            registry: 源注册表。缺省时自动发现 ``plugins/intel_sources`` 下的插件。
            dictionaries: 标的词典，用于实体链接。
            holdings: 当前持仓，影响 importance 与告警。
            watchlist: 候选池。
        """
        self._settings = settings
        self._store = IntelStore(settings.var_dir / "lake" / "intel")
        self._inbox = InboxScanner(settings.var_dir / "intel" / "inbox")
        self._holdings = tuple(holdings)
        self._watchlist = tuple(watchlist)

        # 必须是 `is None` 而不是 `or`：SourceRegistry 定义了 __len__，
        # 空注册表是 falsy。用 `or` 会让"显式关掉所有情报源"被静默替换成
        # 默认装配的联网源——测试里表现为莫名其妙地打真网，
        # 生产上表现为用户关不掉情报采集
        self._registry = (
            registry
            if registry is not None
            else self._build_registry(settings, tracked=(*self._holdings, *self._watchlist))
        )
        self._blacklist = IntelBlacklist()
        self._blacklist.load(self._store.blacklist_path)

        self._pipeline = IntelPipeline(
            linker=EntityLinker(dictionaries, priority=[*self._holdings, *self._watchlist]),
            importance=ImportanceScorer(holdings=self._holdings, watchlist=self._watchlist),
            digest_builder=DigestBuilder(),
            blacklist=self._blacklist,
        )

    @staticmethod
    def _build_registry(settings: Settings, *, tracked: Sequence[Symbol] = ()) -> SourceRegistry:
        """按配置装配情报源。

        插件排在内置源之后注册——同名时用户的实现覆盖内置的，
        这是插件机制该有的优先级。

        Args:
            settings: 运行期配置。
            tracked: 要逐只跟踪个股新闻的标的（持仓 + 候选池）。
                **不传就等于关掉了个股域的采集**——东方财富的个股新闻接口
                是按标的逐个查的，没有标的它只会拉全球快讯。

        Returns:
            源注册表。
        """
        sources: list[NewsSource] = []
        config = settings.config.intel
        if config.enabled:
            # 内置源只在 akshare 可用时才真正工作；不可用时它们的 fetch
            # 返回空列表并记 WARNING，不阻断整体采集
            symbols = tuple(dict.fromkeys(tracked))[:MAX_TRACKED_SYMBOLS]
            if len(tracked) > len(symbols):
                _log.warning(
                    "intel_tracked_symbols_truncated",
                    requested=len(tracked),
                    kept=len(symbols),
                )
            sources.extend([ClsTelegraphSource(), EastMoneySource(symbols=symbols)])
            if feeds := _rss_feeds(settings):
                sources.append(RssSource(feeds))

        sources.extend(discover_plugins(settings.config_dir.parent / "plugins" / "intel_sources"))
        return SourceRegistry(sources)

    @property
    def store(self) -> IntelStore:
        """情报数据湖。"""
        return self._store

    @property
    def registry(self) -> SourceRegistry:
        """源注册表。"""
        return self._registry

    def status(self) -> IntelStatus:
        """当前状态。

        Returns:
            状态快照。
        """
        dates = self._store.available_dates()
        return IntelStatus(
            sources=len(self._registry),
            inbox_pending=sum(1 for _ in self._inbox.pending()),
            latest_date=dates[-1] if dates else None,
            blacklisted=len(self._blacklist.active_entries(as_of=now())),
            dates_available=len(dates),
        )

    def scan_inbox(self, *, move: bool = True) -> ImportReport:
        """扫描外置导入收件箱。

        Args:
            move: 处理后是否移走文件。

        Returns:
            导入报告。
        """
        self._inbox.ensure_dirs()
        return self._inbox.scan(move=move)

    def import_rows(self, rows: Sequence[dict[str, Any]], *, source_name: str) -> list[IntelItem]:
        """导入一批字典（CLI ``intel import`` 与 HTTP 接收端共用）。

        Args:
            rows: 字段字典。
            source_name: 来源名。

        Returns:
            成功解析的条目。
        """
        return items_from_rows(rows, source=f"external:{source_name}")

    def note(self, text: str, **fields: object) -> IntelItem:
        """CLI 直接录入一条情报。

        Args:
            text: 标题或正文。
            **fields: 其它字段，宽松模式，缺省自动补全。

        Returns:
            标准化条目。
        """
        return build_item({"title": text, **fields}, source="external:cli")

    def ingest(
        self, items: Sequence[IntelItem], *, as_of: dt.datetime | None = None
    ) -> PipelineResult:
        """把一批条目走完整流水线后落库。

        **所有入库路径都必须经过这里**——收件箱、CLI 录入、批量导入、HTTP 推送。
        直接往 store 写等于跳过分类、打分与黑名单评估：条目进了库，
        但 ``event_type=None``、``importance=0``，一条"立案调查"公告
        既不会触发禁买、也不会出现在日报的重大消息区。
        这个漏洞在单元测试里看不出来（它们都走 ``fetch``），
        只有端到端跑一遍才会暴露。

        Args:
            items: 待入库条目。
            as_of: 评估时点。

        Returns:
            处理结果。
        """
        result = self._pipeline.process(
            items,
            as_of=as_of or now(),
            holdings=self._holdings,
            watchlist=self._watchlist,
            make_digest=False,
        )
        self._store.write(result.items)
        self._blacklist.save(self._store.blacklist_path, as_of=as_of or now())
        return result

    def fetch(
        self,
        *,
        domains: Sequence[IntelDomain] | None = None,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        include_inbox: bool = True,
        as_of: dt.datetime | None = None,
    ) -> PipelineResult:
        """采集 + 处理 + 落库。

        重复执行幂等：``item_id`` 派生自内容，重复条目在写入时被合并。

        Args:
            domains: 只采集覆盖这些域的源；None 表示全部。
            lookback_days: 向前追溯的天数。
            include_inbox: 是否一并吸收收件箱。
            as_of: 评估时点。回测里传历史时点。

        Returns:
            处理结果。
        """
        moment = as_of or now()
        since = moment - dt.timedelta(days=lookback_days)

        outcome = fetch_all(self._registry, since=since, domains=domains)
        raw: list[IntelItem] = list(outcome.items)
        coverage: dict[str, SourceHealth] = dict(outcome.coverage)

        if include_inbox:
            report = self.scan_inbox()
            raw.extend(report.items)
            coverage["external:inbox"] = SourceHealth(
                source="external:inbox",
                ok=not report.failed,
                fetched=len(report.items),
                error="；".join(reason for _, reason in report.failed),
            )

        result = self._pipeline.process(
            raw,
            as_of=moment,
            holdings=self._holdings,
            watchlist=self._watchlist,
            coverage=coverage,
        )
        self._store.write(result.items)
        if result.digest is not None:
            self._store.save_digest(result.digest)
        self._blacklist.save(self._store.blacklist_path, as_of=moment)
        return result

    def digest(
        self,
        *,
        trade_date: TradeDate | None = None,
        session: str = "post",
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        as_of: dt.datetime | None = None,
    ) -> IntelDigest:
        """按已落库的情报生成摘要（不重新采集）。

        Args:
            trade_date: 交易日。
            session: ``pre`` / ``post``。
            lookback_days: 纳入摘要的回溯天数。
            as_of: PIT 截断时点。

        Returns:
            分域摘要。
        """
        target = trade_date or today()
        moment = as_of or now()
        items = self._store.read_range(
            target - dt.timedelta(days=lookback_days), target, as_of=moment
        )
        digest = DigestBuilder().build(
            items,
            trade_date=target,
            session=session,
            holdings=self._holdings,
            watchlist=self._watchlist,
            generated_at=moment,
        )
        self._store.save_digest(digest)
        return digest

    def render_digest(self, digest: IntelDigest) -> list[str]:
        """渲染摘要为文本行。

        Args:
            digest: 摘要。

        Returns:
            文本行。
        """
        return DigestBuilder.render(digest)

    def blacklist_entries(self, *, as_of: dt.datetime | None = None) -> tuple[BlacklistEntry, ...]:
        """当前生效的黑名单。

        Args:
            as_of: 时点。

        Returns:
            记录元组。
        """
        return self._blacklist.active_entries(as_of=as_of or now())

    def is_blocked(self, symbol: Symbol, *, as_of: dt.datetime | None = None) -> bool:
        """该标的是否被情报禁买。

        Args:
            symbol: 标的。
            as_of: 时点。

        Returns:
            禁买则 True。
        """
        return self._blacklist.is_blocked(symbol, as_of=as_of or now())

    def evidence_for(
        self,
        symbol: Symbol,
        *,
        as_of: dt.datetime | None = None,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        limit: int = 3,
    ) -> list[IntelItem]:
        """取某标的的情报证据，供建议解释的支柱③。

        Args:
            symbol: 标的。
            as_of: PIT 截断时点。
            lookback_days: 回溯天数。
            limit: 最多返回条数。

        Returns:
            按 importance 排序的条目。**每条都带原文链接与发布时间**（红线 I-R4）。
        """
        moment = as_of or now()
        end = moment.date()
        items = self._store.read_range(end - dt.timedelta(days=lookback_days), end, as_of=moment)
        hits = [i for i in items if symbol in i.symbols]
        return sorted(hits, key=lambda i: (-i.importance, i.publish_at))[:limit]

    def absent_note(self, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> str:
        """无相关情报时的说明文字。

        必须明写而不是留空——留空让人分不清"没查"和"查了没有"。

        Args:
            lookback_days: 回溯天数。

        Returns:
            说明文字。
        """
        return f"近 {lookback_days} 日无相关消息"

    @property
    def inbox_dir(self) -> Path:
        """收件箱目录路径，供界面展示。"""
        return self._inbox.inbox_dir


def _rss_feeds(settings: Settings) -> list[RssFeed]:
    """读取用户声明的 RSS 订阅源。

    文件不存在或格式非法都只记警告——RSS 是可选增强，
    一个写坏的 YAML 不该让整个情报模块起不来。

    Args:
        settings: 运行期配置。

    Returns:
        订阅源列表。
    """
    path = settings.config_dir / "intel_sources.yaml"
    if not path.exists():
        return []

    import yaml  # noqa: PLC0415 - 仅此处需要

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        _log.warning("intel_sources_yaml_invalid", path=str(path), error=str(exc))
        return []

    out: list[RssFeed] = []
    for row in payload.get("rss", []):
        if not isinstance(row, dict) or not row.get("url"):
            continue
        domain = IntelDomain.MARKET
        raw_domain = str(row.get("domain", "")).strip().lower()
        for member in IntelDomain:
            if member.value == raw_domain:
                domain = member
                break
        out.append(
            RssFeed(
                name=str(row.get("name") or row["url"]),
                url=str(row["url"]),
                domain=domain,
            )
        )
    return out
