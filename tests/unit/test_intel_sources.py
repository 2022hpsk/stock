"""内置情报源适配器测试（RSS / 财联社 / 东方财富）。

**测试全程不联网**：三个适配器都把第三方库的导入收在 ``_module()`` 里，
测试替换掉这个方法即可用假模块驱动整条解析路径。真实网络用例另行标记
``@pytest.mark.network``，CI 永不执行（docs/01 规范：适配器测试禁止打真实网络）。

重点覆盖的是 PIT 相关的两条底线，它们错了在回测里是**静默**的：

- 没有发布时间的条目必须**丢弃**，不能用抓取时刻顶替（红线 I-R5）；
- feedparser 给的是 UTC ``struct_time``，直接当本地时间会整体偏移 8 小时，
  盘中消息会被算成前一日盘后。
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import IntelError
from quantstock.infra.types import Symbol
from quantstock.intel.protocols import NewsSource
from quantstock.intel.sources.akshare_feeds import ClsTelegraphSource, EastMoneySource
from quantstock.intel.sources.rss import RssFeed, RssSource
from quantstock.intel.types import IntelDomain, SourceTier

FROZEN = dt.datetime(2026, 7, 26, 15, 0, tzinfo=CST)
SINCE = dt.datetime(2026, 7, 20, 0, 0, tzinfo=CST)


@pytest.fixture(autouse=True)
def _frozen_clock() -> None:
    """冻结时钟（红线 R3：禁止依赖真实当前时间）。"""
    set_clock(FrozenClock(FROZEN))


def _utc_struct(moment: dt.datetime) -> time.struct_time:
    """构造 feedparser 风格的 UTC ``struct_time``。

    Args:
        moment: tz-aware 时间。

    Returns:
        UTC 时间元组，与 feedparser 的 ``published_parsed`` 同构。
    """
    return moment.astimezone(dt.UTC).timetuple()


def _entry(
    *,
    title: str = "某公司披露中标公告",
    summary: str = "中标金额 3.2 亿元",
    link: str = "https://example.com/a",
    published: dt.datetime | None = None,
) -> SimpleNamespace:
    """构造一个 feed 条目。

    Args:
        title: 标题。
        summary: 摘要。
        link: 链接。
        published: 发布时间；None 表示 feed 没给日期。

    Returns:
        条目对象。
    """
    entry = SimpleNamespace(title=title, summary=summary, link=link)
    if published is not None:
        entry.published_parsed = _utc_struct(published)
    return entry


class _FakeFeedparser:
    """假的 feedparser 模块。"""

    def __init__(self, mapping: dict[str, Any]) -> None:
        """初始化。

        Args:
            mapping: url → 解析结果或异常。
        """
        self._mapping = mapping
        self.calls: list[str] = []

    def parse(self, url: str) -> Any:
        """解析。

        Args:
            url: 订阅地址。

        Returns:
            解析结果。

        Raises:
            Exception: mapping 里配置的异常。
        """
        self.calls.append(url)
        result = self._mapping[url]
        if isinstance(result, Exception):
            raise result
        return result


def _rss(mapping: dict[str, Any], feeds: list[RssFeed]) -> tuple[RssSource, _FakeFeedparser]:
    """构造注入了假模块的 RSS 源。

    Args:
        mapping: url → 解析结果。
        feeds: 订阅声明。

    Returns:
        (源, 假模块)。
    """
    fake = _FakeFeedparser(mapping)
    source = RssSource(feeds)
    source._module = lambda: fake  # type: ignore[method-assign]
    return source, fake


class TestRssSource:
    """RSS 适配器。"""

    def test_satisfies_news_source_protocol(self) -> None:
        """能被注册表当作情报源使用。"""
        source = RssSource([RssFeed(name="a", url="https://example.com/a.xml")])
        assert isinstance(source, NewsSource)

    def test_utc_struct_time_converted_to_cst(self) -> None:
        """UTC 时间元组必须按 UTC 解释再转 CST，不能当本地时间。

        直接把 ``struct_time`` 当本地时间会整体偏移 8 小时——
        一条 10:30 的盘中消息会变成 02:30，PIT 过滤时归到前一日。
        """
        published = dt.datetime(2026, 7, 24, 10, 30, tzinfo=CST)
        feed = RssFeed(name="a", url="https://example.com/a.xml")
        source, _ = _rss({feed.url: SimpleNamespace(entries=[_entry(published=published)])}, [feed])

        items = source.fetch(SINCE)

        assert len(items) == 1
        assert items[0].publish_at == published

    def test_entry_without_date_is_discarded(self) -> None:
        """没有发布时间的条目丢弃，绝不用抓取时刻顶替（红线 I-R5）。"""
        feed = RssFeed(name="a", url="https://example.com/a.xml")
        source, _ = _rss(
            {
                feed.url: SimpleNamespace(
                    entries=[
                        _entry(published=None, link="https://example.com/nodate"),
                        _entry(published=dt.datetime(2026, 7, 24, 9, 0, tzinfo=CST)),
                    ]
                )
            },
            [feed],
        )

        items = source.fetch(SINCE)

        assert [i.url for i in items] == ["https://example.com/a"]

    def test_updated_parsed_is_used_as_fallback(self) -> None:
        """没有 ``published_parsed`` 时回退 ``updated_parsed``。"""
        moment = dt.datetime(2026, 7, 23, 8, 0, tzinfo=CST)
        entry = SimpleNamespace(title="标题", summary="正文", link="https://example.com/u")
        entry.updated_parsed = _utc_struct(moment)
        feed = RssFeed(name="a", url="https://example.com/a.xml")
        source, _ = _rss({feed.url: SimpleNamespace(entries=[entry])}, [feed])

        items = source.fetch(SINCE)

        assert len(items) == 1
        assert items[0].publish_at == moment

    def test_items_before_since_are_filtered(self) -> None:
        """早于 since 的条目不返回。"""
        feed = RssFeed(name="a", url="https://example.com/a.xml")
        source, _ = _rss(
            {
                feed.url: SimpleNamespace(
                    entries=[
                        _entry(
                            published=dt.datetime(2026, 7, 1, 9, 0, tzinfo=CST),
                            link="https://example.com/old",
                        ),
                        _entry(published=dt.datetime(2026, 7, 25, 9, 0, tzinfo=CST)),
                    ]
                )
            },
            [feed],
        )

        items = source.fetch(SINCE)

        assert [i.url for i in items] == ["https://example.com/a"]

    def test_one_broken_feed_does_not_kill_the_rest(self) -> None:
        """单个 feed 失败只丢那一个。

        情报是增强项不是前置条件：一个订阅源挂掉不该让当天所有情报都拿不到。
        """
        ok = RssFeed(name="ok", url="https://example.com/ok.xml")
        bad = RssFeed(name="bad", url="https://example.com/bad.xml")
        source, _ = _rss(
            {
                bad.url: RuntimeError("connection reset"),
                ok.url: SimpleNamespace(
                    entries=[_entry(published=dt.datetime(2026, 7, 24, 9, 0, tzinfo=CST))]
                ),
            },
            [bad, ok],
        )

        items = source.fetch(SINCE)

        assert len(items) == 1
        assert items[0].source == "rss:ok"

    def test_feed_declares_domain_and_tier(self) -> None:
        """条目继承订阅源声明的域与层级，且标记为源方声明。"""
        feed = RssFeed(
            name="gov",
            url="https://example.com/gov.xml",
            domain=IntelDomain.POLICY,
            tier=SourceTier.OFFICIAL,
        )
        source, _ = _rss({feed.url: SimpleNamespace(entries=[_entry(published=FROZEN)])}, [feed])

        item = source.fetch(SINCE)[0]

        assert item.domain is IntelDomain.POLICY
        assert item.source_tier is SourceTier.OFFICIAL
        assert item.domain_declared is True

    def test_item_id_is_idempotent(self) -> None:
        """同一条消息重复抓取落到同一个 id。"""
        feed = RssFeed(name="a", url="https://example.com/a.xml")
        parsed = SimpleNamespace(entries=[_entry(published=FROZEN)])
        source, _ = _rss({feed.url: parsed}, [feed])

        first = source.fetch(SINCE)[0]
        second = source.fetch(SINCE)[0]

        assert first.item_id == second.item_id

    def test_domains_deduplicates(self) -> None:
        """多个同域订阅源只报告一次该域。"""
        source = RssSource(
            [
                RssFeed(name="a", url="https://a", domain=IntelDomain.MARKET),
                RssFeed(name="b", url="https://b", domain=IntelDomain.MARKET),
                RssFeed(name="c", url="https://c", domain=IntelDomain.POLICY),
            ]
        )

        assert source.domains == (IntelDomain.MARKET, IntelDomain.POLICY)

    def test_health_ok_when_any_feed_alive(self) -> None:
        """只要有一个订阅源可达就算可用。"""
        ok = RssFeed(name="ok", url="https://example.com/ok.xml")
        bad = RssFeed(name="bad", url="https://example.com/bad.xml")
        source, _ = _rss(
            {
                bad.url: RuntimeError("timeout"),
                ok.url: SimpleNamespace(entries=[_entry(published=FROZEN)]),
            },
            [bad, ok],
        )

        health = source.health_check()

        assert health.ok is True
        assert health.fetched == 1

    def test_health_reports_total_outage(self) -> None:
        """全部不可达时报告失败原因。"""
        bad = RssFeed(name="bad", url="https://example.com/bad.xml")
        source, _ = _rss({bad.url: RuntimeError("timeout")}, [bad])

        health = source.health_check()

        assert health.ok is False
        assert "不可达" in health.error

    def test_health_without_feeds(self) -> None:
        """没配订阅源时明确说明，而不是假装健康。"""
        health = RssSource([]).health_check()

        assert health.ok is False
        assert "未配置" in health.error

    def test_missing_dependency_raises_intel_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未安装 feedparser 时给出可操作的错误信息。

        ``sys.modules`` 里放 None 会让 import 抛 ``ImportError``，
        以此模拟"没装这个包"而不必真的卸载它。
        """
        monkeypatch.setitem(sys.modules, "feedparser", None)
        source = RssSource([RssFeed(name="a", url="https://a")])

        with pytest.raises(IntelError, match="feedparser"):
            source._module()


class _FakeFrame:
    """假的 pandas DataFrame（只用到 ``len`` 与 ``to_dict``）。"""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """初始化。

        Args:
            rows: 数据行。
        """
        self._rows = rows

    def __len__(self) -> int:
        """行数。"""
        return len(self._rows)

    def to_dict(self, orient: str) -> list[dict[str, Any]]:
        """转字典。

        Args:
            orient: 方向，只支持 ``records``。

        Returns:
            数据行。
        """
        assert orient == "records"
        return self._rows


class _FakeAkShare:
    """假的 akshare 模块。"""

    def __init__(self, **endpoints: Any) -> None:
        """初始化。

        Args:
            endpoints: 接口名 → 返回值或异常。
        """
        self._endpoints = endpoints
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        """按名取接口。

        Args:
            name: 接口名。

        Returns:
            可调用对象。

        Raises:
            AttributeError: 未配置该接口。
        """
        if name not in self._endpoints:
            msg = f"fake akshare has no endpoint {name}"
            raise AttributeError(msg)

        def _call(**kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            result = self._endpoints[name]
            if isinstance(result, Exception):
                raise result
            return result

        return _call


def _inject(source: Any, module: _FakeAkShare) -> None:
    """把假模块注入适配器。

    Args:
        source: 适配器实例。
        module: 假 akshare 模块。
    """
    source._module = lambda: module


class TestClsTelegraphSource:
    """财联社电报。"""

    def test_satisfies_protocol(self) -> None:
        """满足情报源协议。"""
        assert isinstance(ClsTelegraphSource(), NewsSource)

    def test_parses_chinese_columns(self) -> None:
        """中文列名照常解析。"""
        source = ClsTelegraphSource()
        _inject(
            source,
            _FakeAkShare(
                stock_info_global_cls=_FakeFrame(
                    [
                        {
                            "标题": "央行开展 5000 亿元逆回购",
                            "内容": "利率持平于 1.40%",
                            "发布时间": "2026-07-24 09:20:00",
                            "链接": "https://cls.cn/1",
                        }
                    ]
                )
            ),
        )

        items = source.fetch(SINCE)

        assert len(items) == 1
        assert items[0].title == "央行开展 5000 亿元逆回购"
        assert items[0].publish_at == dt.datetime(2026, 7, 24, 9, 20, tzinfo=CST)
        assert items[0].source_tier is SourceTier.MEDIA

    def test_time_only_format_uses_today(self) -> None:
        """电报常只给"时:分"，补今天的日期。

        补昨天或留空都会让 PIT 过滤算错——这类接口返回的本来就是当日快讯。
        """
        source = ClsTelegraphSource()
        _inject(
            source,
            _FakeAkShare(
                stock_info_global_cls=_FakeFrame(
                    [{"标题": "快讯", "内容": "正文", "发布时间": "10:35"}]
                )
            ),
        )

        items = source.fetch(SINCE)

        assert items[0].publish_at == dt.datetime(2026, 7, 26, 10, 35, tzinfo=CST)

    def test_unparseable_date_is_discarded(self) -> None:
        """时间解析不出来就丢弃该条，而不是用当前时刻顶替。"""
        source = ClsTelegraphSource()
        _inject(
            source,
            _FakeAkShare(
                stock_info_global_cls=_FakeFrame(
                    [
                        {"标题": "坏数据", "内容": "x", "发布时间": "刚刚"},
                        {"标题": "好数据", "内容": "y", "发布时间": "2026-07-25 09:00:00"},
                    ]
                )
            ),
        )

        items = source.fetch(SINCE)

        assert [i.title for i in items] == ["好数据"]

    def test_fetch_failure_degrades_to_empty(self) -> None:
        """接口报错时返回空列表而不是抛异常。

        情报缺失只降级为"缺少证据"，不阻断当日建议生成。
        """
        source = ClsTelegraphSource()
        _inject(source, _FakeAkShare(stock_info_global_cls=RuntimeError("502")))

        assert source.fetch(SINCE) == []

    def test_health_check_reports_error(self) -> None:
        """健康检查把异常转成健康度而不是外抛。"""
        source = ClsTelegraphSource()
        _inject(source, _FakeAkShare(stock_info_global_cls=RuntimeError("boom")))

        health = source.health_check()

        assert health.ok is False
        assert "boom" in health.error


class TestEastMoneySource:
    """东方财富。"""

    def test_per_symbol_news_carries_symbol(self) -> None:
        """个股新闻带上标的，供实体链接直接使用。"""
        symbol = Symbol("600519.SH")
        source = EastMoneySource(symbols=[symbol])
        module = _FakeAkShare(
            stock_info_global_em=_FakeFrame([]),
            stock_news_em=_FakeFrame(
                [
                    {
                        "新闻标题": "公司发布半年报",
                        "新闻内容": "营收同比增长 12%",
                        "发布时间": "2026-07-25 18:00:00",
                        "新闻链接": "https://em.com/1",
                    }
                ]
            ),
        )
        _inject(source, module)

        items = source.fetch(SINCE)

        assert len(items) == 1
        assert items[0].symbols == (symbol,)
        assert items[0].domain is IntelDomain.COMPANY
        assert ("stock_news_em", {"symbol": "600519"}) in module.calls

    def test_global_feed_is_market_domain(self) -> None:
        """全球快讯归到市场域。"""
        source = EastMoneySource()
        _inject(
            source,
            _FakeAkShare(
                stock_info_global_em=_FakeFrame(
                    [
                        {
                            "标题": "美联储维持利率不变",
                            "摘要": "点阵图显示年内一次降息",
                            "发布时间": "2026-07-24 22:00:00",
                        }
                    ]
                )
            ),
        )

        items = source.fetch(SINCE)

        assert len(items) == 1
        assert items[0].domain is IntelDomain.MARKET

    def test_one_symbol_failure_does_not_stop_others(self) -> None:
        """某只标的查询失败不影响其余标的。"""
        calls: list[str] = []

        class _Flaky(_FakeAkShare):
            def __getattr__(self, name: str) -> Any:
                if name != "stock_news_em":
                    return super().__getattr__(name)

                def _call(*, symbol: str) -> Any:
                    calls.append(symbol)
                    if symbol == "600519":
                        msg = "rate limited"
                        raise RuntimeError(msg)
                    return _FakeFrame(
                        [
                            {
                                "新闻标题": "标题",
                                "新闻内容": "正文",
                                "发布时间": "2026-07-25 10:00:00",
                            }
                        ]
                    )

                return _call

        source = EastMoneySource(symbols=[Symbol("600519.SH"), Symbol("000001.SZ")])
        _inject(source, _Flaky(stock_info_global_em=_FakeFrame([])))

        items = source.fetch(SINCE)

        assert calls == ["600519", "000001"]
        assert len(items) == 1


class TestEndpointDrift:
    """AkShare 接口改名的兼容。

    ``stock_telegraph_cls`` 在 1.18 已并入 ``stock_info_global_cls``。
    只认一个名字的实现，会在用户升级 AkShare 的那天静默停止采集——
    ``fetch`` 抓不到 AttributeError 就直接崩，抓到了就永远返回空。
    """

    def test_legacy_endpoint_name_still_works(self) -> None:
        source = ClsTelegraphSource()
        _inject(
            source,
            _FakeAkShare(
                stock_telegraph_cls=_FakeFrame(
                    [{"标题": "旧接口", "内容": "x", "发布时间": "2026-07-25 09:00:00"}]
                )
            ),
        )

        assert [i.title for i in source.fetch(SINCE)] == ["旧接口"]

    def test_new_endpoint_wins_when_both_exist(self) -> None:
        source = ClsTelegraphSource()
        _inject(
            source,
            _FakeAkShare(
                stock_info_global_cls=_FakeFrame(
                    [{"标题": "新接口", "内容": "x", "发布时间": "2026-07-25 09:00:00"}]
                ),
                stock_telegraph_cls=_FakeFrame(
                    [{"标题": "旧接口", "内容": "x", "发布时间": "2026-07-25 09:00:00"}]
                ),
            ),
        )

        assert [i.title for i in source.fetch(SINCE)] == ["新接口"]

    def test_no_known_endpoint_degrades_to_empty(self) -> None:
        source = ClsTelegraphSource()
        _inject(source, _FakeAkShare())

        assert source.fetch(SINCE) == []

    def test_health_check_fails_when_endpoint_vanishes(self) -> None:
        # 健康检查必须走与 fetch 相同的候选解析，否则改名后它会说"可用"
        # 而实际一条都采不到
        source = ClsTelegraphSource()
        _inject(source, _FakeAkShare())

        health = source.health_check()

        assert health.ok is False
        assert "无可用接口" in health.error

    def test_eastmoney_skips_per_symbol_when_endpoint_missing(self) -> None:
        source = EastMoneySource(symbols=[Symbol("600519.SH")])
        _inject(source, _FakeAkShare(stock_info_global_em=_FakeFrame([])))

        assert source.fetch(SINCE) == []
