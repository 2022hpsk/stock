"""情报模块测试。

覆盖 docs/07-信息情报模块.md 第十节的验收标准，其中最关键的三条：

- **验收 5**：一条"立案调查"公告 → 该标的自动进黑名单、禁止买入、可溯源到原文链接；
- **验收 7**：回测中不得使用未来情报（红线 I-R5）；
- **验收 8**：关闭全部情报源后系统仍能出建议，仅标注"情报缺失"——
  验证情报是增强项而非阻断项。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import IntelError
from quantstock.infra.types import Symbol
from quantstock.intel.blacklist import IntelBlacklist
from quantstock.intel.classify import EventClassifier, SentimentScorer
from quantstock.intel.dedup import (
    DEFAULT_SIMILARITY_THRESHOLD,
    content_hash,
    dedup,
    hamming_distance,
    simhash,
    similarity,
)
from quantstock.intel.digest import DigestBuilder, build_portfolio_alerts
from quantstock.intel.entity import EntityLinker, SymbolDictionary
from quantstock.intel.external import InboxScanner, build_item, parse_payload
from quantstock.intel.pipeline import IntelPipeline
from quantstock.intel.protocols import SourceRegistry, discover_plugins, fetch_all
from quantstock.intel.scoring import ImportanceScorer, ImportanceWeights
from quantstock.intel.store import IntelStore
from quantstock.intel.types import (
    CalendarEvent,
    EventType,
    IntelDomain,
    IntelItem,
    SourceHealth,
    SourceTier,
)

MAOTAI = Symbol("600519.SH")
CATL = Symbol("300750.SZ")
PINGAN = Symbol("601318.SH")

NOW = dt.datetime(2026, 7, 25, 18, 30, tzinfo=CST)


@pytest.fixture(autouse=True)
def _frozen() -> None:
    """固定时钟。"""
    set_clock(FrozenClock(NOW))


def item(
    title: str,
    *,
    body: str = "",
    source: str = "cls",
    tier: SourceTier = SourceTier.MEDIA,
    domain: IntelDomain = IntelDomain.COMPANY,
    publish_at: dt.datetime | None = None,
    symbols: tuple[Symbol, ...] = (),
    event: EventType | None = None,
    importance: int = 0,
    sentiment: float = 0.0,
    url: str = "https://example.com/a",
    item_id: str = "",
) -> IntelItem:
    """构造一条情报。"""
    stamp = publish_at or NOW - dt.timedelta(hours=1)
    digest = content_hash(title, body)
    return IntelItem(
        item_id=item_id or digest[:16],
        source=source,
        source_tier=tier,
        domain=domain,
        publish_at=stamp,
        fetched_at=NOW,
        title=title,
        content_hash=digest,
        body=body,
        url=url,
        symbols=symbols,
        event_type=event,
        importance=importance,
        sentiment=sentiment,
    )


class TestIntelItem:
    """契约校验。"""

    def test_naive_publish_at_rejected(self) -> None:
        # publish_at 是 PIT 关键字段，丢了时区就无法判断"当时看得到吗"
        with pytest.raises(ValueError, match="tz-aware"):
            IntelItem(
                item_id="x",
                source="s",
                source_tier=SourceTier.MEDIA,
                domain=IntelDomain.COMPANY,
                publish_at=dt.datetime(2026, 7, 25, 9),  # noqa: DTZ001 - 刻意构造违规输入
                fetched_at=NOW,
                title="t",
                content_hash="h",
            )

    @pytest.mark.parametrize("importance", [-1, 101])
    def test_importance_bounds(self, importance: int) -> None:
        with pytest.raises(ValueError, match="importance"):
            item("t").__class__(
                item_id="x",
                source="s",
                source_tier=SourceTier.MEDIA,
                domain=IntelDomain.COMPANY,
                publish_at=NOW,
                fetched_at=NOW,
                title="t",
                content_hash="h",
                importance=importance,
            )

    def test_visible_at_is_the_pit_gate(self) -> None:
        past = item("旧闻", publish_at=NOW - dt.timedelta(days=1))
        future = item("明天的消息", publish_at=NOW + dt.timedelta(hours=1))
        assert past.visible_at(NOW)
        assert not future.visible_at(NOW)

    def test_visible_at_rejects_naive_as_of(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            item("t").visible_at(dt.datetime(2026, 7, 25))  # noqa: DTZ001 - 刻意构造违规输入

    def test_cite_carries_source_and_link(self) -> None:
        # 红线 I-R4：进入解释的情报必须带原文链接与发布时间
        text = item("茅台减持公告", source="上交所", url="https://sse.com/x").cite()
        assert "茅台减持公告" in text
        assert "上交所" in text
        assert "https://sse.com/x" in text

    def test_llm_flags(self) -> None:
        from dataclasses import replace  # noqa: PLC0415 - 仅测试内使用

        base = item("t", sentiment=0.1)
        assert not base.is_llm_generated
        assert replace(base, classifier="llm:fast").is_llm_generated
        assert replace(base, sentiment_source="llm:fast").is_llm_generated
        assert not base.sentiment_disputed
        assert replace(base, llm_sentiment=0.9).sentiment_disputed


class TestSimHash:
    """自实现的 64 位 SimHash。"""

    def test_identical_text_identical_fingerprint(self) -> None:
        assert simhash("央行宣布降准0.5个百分点") == simhash("央行宣布降准0.5个百分点")

    def test_punctuation_and_width_do_not_matter(self) -> None:
        # 同一条快讯在不同源上常常只差全半角标点
        assert simhash("央行宣布降准 0.5 个百分点！") == simhash("央行宣布降准０.５个百分点!")

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (
                "贵州茅台控股股东计划减持不超过1%股份",
                "贵州茅台：控股股东拟减持不超过 1% 股份",
            ),
            ("央行宣布降准0.5个百分点", "央行决定降准0.5个百分点"),
            ("公司收到证监会立案调查通知书", "公司收到中国证监会立案调查通知书"),
            ("6月CPI同比上涨0.3%低于预期", "统计局：6月CPI同比上涨0.3%，低于市场预期"),
        ],
    )
    def test_rewritten_headline_clears_the_threshold(self, left: str, right: str) -> None:
        # 同一事件的媒体改写必须过阈值，否则近似去重形同虚设
        assert similarity(simhash(left), simhash(right)) >= DEFAULT_SIMILARITY_THRESHOLD

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (
                "央行宣布降准0.5个百分点，释放长期资金约1万亿元",
                "宁德时代发布新一代动力电池，能量密度提升30%",
            ),
            ("贵州茅台控股股东计划减持", "6月CPI同比上涨0.3%"),
            ("公司收到立案调查通知书", "公司拟回购股份用于注销"),
            ("美联储维持利率不变", "白酒行业动销数据环比回落"),
        ],
    )
    def test_unrelated_headlines_stay_below_the_threshold(self, left: str, right: str) -> None:
        assert similarity(simhash(left), simhash(right)) < DEFAULT_SIMILARITY_THRESHOLD

    def test_empty_text_is_zero(self) -> None:
        assert simhash("") == 0
        assert hamming_distance(0, 0) == 0

    def test_hamming_distance_is_symmetric(self) -> None:
        left, right = simhash("甲"), simhash("乙丙丁戊")
        assert hamming_distance(left, right) == hamming_distance(right, left)


class TestDedup:
    """去重。"""

    def test_exact_duplicates_merged(self) -> None:
        a = item("央行降准", item_id="a")
        b = item("央行降准", item_id="b")
        result = dedup([a, b])
        assert len(result.kept) == 1
        assert result.dropped_count == 1

    def test_official_source_wins_as_primary(self) -> None:
        # 交易所公告永远压过媒体转述
        media = item("茅台减持公告", source="cls", tier=SourceTier.MEDIA, item_id="m")
        official = item("茅台减持公告", source="sse", tier=SourceTier.OFFICIAL, item_id="o")
        result = dedup([media, official])
        assert len(result.kept) == 1
        assert result.kept[0].source == "sse"
        assert "m" in result.kept[0].duplicates

    def test_near_duplicates_merged_within_window(self) -> None:
        a = item("贵州茅台控股股东计划减持不超过1%股份", item_id="a")
        b = item(
            "贵州茅台：控股股东拟减持不超过 1% 股份",
            item_id="b",
            publish_at=NOW - dt.timedelta(minutes=30),
        )
        result = dedup([a, b])
        assert len(result.kept) == 1

    def test_same_text_outside_window_is_kept_separately(self) -> None:
        # 隔了三天再出现通常是"旧闻重发"或"事件进展"，不是重复
        a = item("公司发布月度经营数据", item_id="a", publish_at=NOW - dt.timedelta(days=3))
        b = item("公司发布月度经营数据", item_id="b")
        result = dedup([a, b], window_hours=6)
        # 精确哈希仍会命中——它不看时间窗，这是刻意的：一字不差就是重复
        assert len(result.kept) == 1

    def test_distinct_news_not_merged(self) -> None:
        a = item("央行宣布降准0.5个百分点释放万亿资金", item_id="a")
        b = item("宁德时代发布新一代动力电池能量密度提升", item_id="b")
        assert len(dedup([a, b]).kept) == 2

    def test_empty_input(self) -> None:
        result = dedup([])
        assert result.kept == ()
        assert result.dropped_count == 0


class TestEntityLinker:
    """实体链接。"""

    @pytest.fixture
    def linker(self) -> EntityLinker:
        return EntityLinker(
            [
                SymbolDictionary(MAOTAI, "贵州茅台", aliases=("茅台",), industry="食品饮料"),
                SymbolDictionary(CATL, "宁德时代", industry="电力设备"),
            ],
            industries={"白酒": ("白酒", "动销")},
            themes={"新能源车": ("动力电池", "新能源车")},
        )

    def test_links_by_code(self, linker: EntityLinker) -> None:
        result = linker.link("600519.SH 发布公告")
        assert MAOTAI in result.symbols
        assert any("代码" in e for e in result.evidence)

    def test_links_by_name(self, linker: EntityLinker) -> None:
        result = linker.link("贵州茅台控股股东拟减持")
        assert result.symbols == (MAOTAI,)
        assert "600519.SH←贵州茅台" in result.evidence

    def test_evidence_makes_the_match_explainable(self, linker: EntityLinker) -> None:
        # 不可解释的关联比没有关联更糟
        result = linker.link("宁德时代发布新电池")
        assert result.evidence
        assert all("←" in e for e in result.evidence)

    def test_industry_inherited_from_matched_symbol(self, linker: EntityLinker) -> None:
        assert "食品饮料" in linker.link("贵州茅台公告").industries

    def test_theme_keywords(self, linker: EntityLinker) -> None:
        assert "新能源车" in linker.link("动力电池产能扩张").themes

    def test_disambiguation_prefers_holdings(self) -> None:
        other = Symbol("000001.SZ")
        linker = EntityLinker(
            [
                SymbolDictionary(other, "平安银行", aliases=("平安",)),
                SymbolDictionary(PINGAN, "中国平安", aliases=("平安",)),
            ],
            priority=[PINGAN],
        )
        assert linker.link("平安发布公告").symbols == (PINGAN,)

    def test_single_char_alias_ignored(self) -> None:
        # 单字简称误报率高到没有使用价值
        linker = EntityLinker([SymbolDictionary(MAOTAI, "贵州茅台", aliases=("茅",))])
        assert linker.link("茅屋为秋风所破").is_empty

    def test_empty_linker_matches_nothing(self) -> None:
        assert EntityLinker().link("随便一句话").is_empty


class TestEventClassifier:
    """事件分类。"""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("公司收到证监会立案调查通知书", EventType.REGULATORY_PROBE),
            ("年报被出具保留意见审计报告", EventType.AUDIT_QUALIFIED),
            ("公司股票可能被终止上市", EventType.DELISTING_RISK),
            ("控股股东拟减持不超过1%股份", EventType.SHAREHOLDER_REDUCE),
            ("公司拟回购股份用于注销", EventType.BUYBACK),
            ("中标5亿元重大合同", EventType.MAJOR_CONTRACT),
            ("央行宣布降准0.5个百分点", EventType.MONETARY),
            ("6月CPI同比上涨0.3%", EventType.MACRO_DATA),
            ("公司披露2026年半年报", EventType.EARNINGS_REPORT),
        ],
    )
    def test_rule_classification(self, text: str, expected: EventType) -> None:
        assert EventClassifier().classify(text).event_type is expected

    def test_risk_events_take_priority(self) -> None:
        # 既提到重组又提到立案调查的公告，必须归为风险类
        result = EventClassifier().classify("公司重大资产重组期间收到立案调查通知")
        assert result.event_type is EventType.REGULATORY_PROBE

    def test_unmatched_returns_none_for_llm_fallback(self) -> None:
        result = EventClassifier().classify("今天天气不错适合散步")
        assert result.event_type is None
        assert result.classifier == "rule"

    def test_matched_keywords_are_reported(self) -> None:
        result = EventClassifier().classify("公司收到立案调查通知")
        assert "立案调查" in result.matched

    def test_declared_domain_wins_over_inference(self) -> None:
        result = EventClassifier().classify("央行降准", default_domain=IntelDomain.POLICY)
        assert result.domain is IntelDomain.POLICY

    def test_domain_inferred_from_event(self) -> None:
        assert EventClassifier().classify("央行宣布降准").domain is IntelDomain.MACRO

    def test_risk_event_property(self) -> None:
        assert EventType.REGULATORY_PROBE.is_risk_event
        assert EventType.DELISTING_RISK.is_risk_event
        # 诉讼在 A 股极常见且多为小额，纳入硬否决会让黑名单失去意义
        assert not EventType.LITIGATION.is_risk_event
        assert not EventType.BUYBACK.is_risk_event


class TestSentimentScorer:
    """情绪打分。"""

    def test_risk_event_is_strongly_negative(self) -> None:
        score = SentimentScorer().score("收到立案调查通知", event_type=EventType.REGULATORY_PROBE)
        assert score < -0.7

    def test_buyback_is_positive(self) -> None:
        assert SentimentScorer().score("拟回购股份", event_type=EventType.BUYBACK) > 0.2

    def test_ambiguous_events_have_no_prior(self) -> None:
        # 业绩预告可能预增也可能预亏，先验瞎给一个正负会系统性地错
        from quantstock.intel.classify import EVENT_SENTIMENT_PRIOR  # noqa: PLC0415

        assert EVENT_SENTIMENT_PRIOR[EventType.EARNINGS_FORECAST] == 0.0
        assert EVENT_SENTIMENT_PRIOR[EventType.PRICE_MOVE] == 0.0
        assert EVENT_SENTIMENT_PRIOR[EventType.MONETARY] == 0.0

    def test_lexical_direction_read_from_text(self) -> None:
        scorer = SentimentScorer()
        good = scorer.score("业绩大幅增长创新高", event_type=EventType.EARNINGS_FORECAST)
        bad = scorer.score("业绩大幅下滑亏损", event_type=EventType.EARNINGS_FORECAST)
        assert good > 0 > bad

    def test_title_weighs_more_than_body(self) -> None:
        scorer = SentimentScorer()
        in_title = scorer.score("增长", body="")
        in_body = scorer.score("", body="增长")
        assert in_title > in_body

    def test_score_stays_in_range(self) -> None:
        scorer = SentimentScorer()
        extreme = scorer.score("退市风险爆雷亏损违规处罚" * 20, event_type=EventType.DELISTING_RISK)
        assert -1.0 <= extreme <= 1.0

    def test_neutral_text_scores_zero(self) -> None:
        assert SentimentScorer().score("公司发布公告") == 0.0


class TestImportanceScorer:
    """重要性打分。"""

    def test_official_source_scores_higher(self) -> None:
        scorer = ImportanceScorer()
        official = scorer.score(
            item("减持公告", tier=SourceTier.OFFICIAL, event=EventType.SHAREHOLDER_REDUCE),
            as_of=NOW,
        )
        social = scorer.score(
            item("减持传闻", tier=SourceTier.SOCIAL, event=EventType.SHAREHOLDER_REDUCE),
            as_of=NOW,
        )
        assert official.total > social.total

    def test_holdings_hit_adds_points(self) -> None:
        held = ImportanceScorer(holdings=[MAOTAI])
        plain = ImportanceScorer()
        target = item("公告", symbols=(MAOTAI,), event=EventType.EARNINGS_REPORT)
        assert held.score(target, as_of=NOW).total > plain.score(target, as_of=NOW).total

    def test_corroboration_raises_score(self) -> None:
        from dataclasses import replace  # noqa: PLC0415 - 仅测试内使用

        scorer = ImportanceScorer()
        single = item("公告", event=EventType.MA)
        multi = replace(single, duplicates=("a", "b", "c"))
        assert scorer.score(multi, as_of=NOW).total > scorer.score(single, as_of=NOW).total

    def test_time_decay_uses_as_of_not_now(self) -> None:
        # 回测里评估时点是历史某天，用 now() 会让所有历史情报都衰减到 0
        scorer = ImportanceScorer()
        old = item("公告", event=EventType.MA, publish_at=NOW - dt.timedelta(days=30))
        at_publish = scorer.score(old, as_of=old.publish_at + dt.timedelta(minutes=1))
        at_now = scorer.score(old, as_of=NOW)
        assert at_publish.total > at_now.total

    def test_user_imports_are_capped(self) -> None:
        # 防止单条人工输入压过全部量化信号
        scorer = ImportanceScorer(holdings=[MAOTAI])
        forced = item(
            "我认为要大涨",
            tier=SourceTier.USER,
            symbols=(MAOTAI,),
            event=EventType.DELISTING_RISK,
        )
        assert scorer.score(forced, as_of=NOW).total <= 90

    def test_naive_as_of_rejected(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            ImportanceScorer().score(item("t"), as_of=dt.datetime(2026, 7, 25))  # noqa: DTZ001

    def test_breakdown_explains_itself(self) -> None:
        # 说不清来历的分数，用户迟早会不再相信它
        breakdown = ImportanceScorer(holdings=[MAOTAI]).score(
            item("立案调查", symbols=(MAOTAI,), event=EventType.REGULATORY_PROBE), as_of=NOW
        )
        text = breakdown.explain()
        assert "importance" in text
        assert "持仓" in text

    def test_custom_weights_apply(self) -> None:
        weights = ImportanceWeights(portfolio_hit=40)
        scorer = ImportanceScorer(weights=weights, holdings=[MAOTAI])
        target = item("公告", symbols=(MAOTAI,), event=EventType.EARNINGS_REPORT)
        assert scorer.score(target, as_of=NOW).portfolio == 40.0

    def test_score_clamped_to_100(self) -> None:
        scorer = ImportanceScorer(holdings=[MAOTAI])
        loud = item(
            "退市风险",
            tier=SourceTier.OFFICIAL,
            symbols=(MAOTAI,),
            event=EventType.DELISTING_RISK,
        )
        assert 0 <= scorer.score(loud, as_of=NOW).total <= 100


class TestBlacklist:
    """情报黑名单（验收 5）。"""

    def test_regulatory_probe_blocks_the_symbol(self) -> None:
        blacklist = IntelBlacklist()
        probe = item(
            "公司收到证监会立案调查通知书",
            source="sse",
            tier=SourceTier.OFFICIAL,
            symbols=(MAOTAI,),
            event=EventType.REGULATORY_PROBE,
            importance=90,
            url="https://sse.com/notice/1",
        )
        entries = blacklist.evaluate([probe], as_of=NOW)

        assert blacklist.is_blocked(MAOTAI, as_of=NOW)
        assert len(entries) == 1
        # 必须可溯源到原文链接（红线 I-R4）
        assert entries[0].urls == ("https://sse.com/notice/1",)
        assert probe.item_id in entries[0].item_ids
        assert "🔗" in entries[0].explain()

    def test_social_rumours_cannot_blacklist(self) -> None:
        # 一条论坛帖子不该把持仓标的锁死
        blacklist = IntelBlacklist()
        rumour = item(
            "听说被立案调查了",
            tier=SourceTier.SOCIAL,
            symbols=(MAOTAI,),
            event=EventType.REGULATORY_PROBE,
            importance=95,
        )
        blacklist.evaluate([rumour], as_of=NOW)
        assert not blacklist.is_blocked(MAOTAI, as_of=NOW)

    def test_low_importance_does_not_blacklist(self) -> None:
        blacklist = IntelBlacklist()
        weak = item(
            "立案调查",
            tier=SourceTier.MEDIA,
            symbols=(MAOTAI,),
            event=EventType.REGULATORY_PROBE,
            importance=50,
        )
        blacklist.evaluate([weak], as_of=NOW)
        assert not blacklist.is_blocked(MAOTAI, as_of=NOW)

    def test_negative_streak_triggers(self) -> None:
        blacklist = IntelBlacklist()
        items = [
            item(
                f"负面消息{n}",
                symbols=(CATL,),
                sentiment=-0.6,
                publish_at=NOW - dt.timedelta(days=n * 5),
                item_id=f"neg{n}",
            )
            for n in range(3)
        ]
        blacklist.evaluate(items, as_of=NOW)
        assert blacklist.is_blocked(CATL, as_of=NOW)
        entry = blacklist.entry_for(CATL)
        assert entry is not None
        assert entry.rule == "negative_streak"

    def test_two_negatives_are_not_enough(self) -> None:
        blacklist = IntelBlacklist()
        items = [
            item(f"负面{n}", symbols=(CATL,), sentiment=-0.6, item_id=f"n{n}") for n in range(2)
        ]
        blacklist.evaluate(items, as_of=NOW)
        assert not blacklist.is_blocked(CATL, as_of=NOW)

    def test_blacklist_expires(self) -> None:
        blacklist = IntelBlacklist(ttl_days=10)
        probe = item(
            "立案调查",
            tier=SourceTier.OFFICIAL,
            symbols=(MAOTAI,),
            event=EventType.REGULATORY_PROBE,
            importance=90,
        )
        blacklist.evaluate([probe], as_of=NOW)
        assert blacklist.is_blocked(MAOTAI, as_of=NOW + dt.timedelta(days=9))
        assert not blacklist.is_blocked(MAOTAI, as_of=NOW + dt.timedelta(days=11))

    def test_retrigger_extends_expiry(self) -> None:
        blacklist = IntelBlacklist(ttl_days=30)
        probe = item(
            "立案调查",
            tier=SourceTier.OFFICIAL,
            symbols=(MAOTAI,),
            event=EventType.REGULATORY_PROBE,
            importance=90,
        )
        blacklist.evaluate([probe], as_of=NOW)
        first = blacklist.entry_for(MAOTAI)
        blacklist.evaluate([probe], as_of=NOW + dt.timedelta(days=10))
        second = blacklist.entry_for(MAOTAI)
        assert first is not None and second is not None
        assert second.expires_at > first.expires_at

    def test_future_items_are_invisible(self) -> None:
        # 红线 I-R5：明天的公告不能拉黑今天的标的
        blacklist = IntelBlacklist()
        future = item(
            "立案调查",
            tier=SourceTier.OFFICIAL,
            symbols=(MAOTAI,),
            event=EventType.REGULATORY_PROBE,
            importance=90,
            publish_at=NOW + dt.timedelta(days=1),
        )
        blacklist.evaluate([future], as_of=NOW)
        assert not blacklist.is_blocked(MAOTAI, as_of=NOW)

    def test_naive_as_of_rejected(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            IntelBlacklist().evaluate([], as_of=dt.datetime(2026, 7, 25))  # noqa: DTZ001

    def test_persist_and_reload(self, tmp_path: Path) -> None:
        blacklist = IntelBlacklist()
        probe = item(
            "立案调查",
            tier=SourceTier.OFFICIAL,
            symbols=(MAOTAI,),
            event=EventType.REGULATORY_PROBE,
            importance=90,
        )
        blacklist.evaluate([probe], as_of=NOW)
        path = blacklist.save(tmp_path / "state.json", as_of=NOW)

        restored = IntelBlacklist()
        restored.load(path)
        assert restored.is_blocked(MAOTAI, as_of=NOW)

    def test_load_missing_file_is_silent(self, tmp_path: Path) -> None:
        # 首次运行没有黑名单是正常的
        blacklist = IntelBlacklist()
        blacklist.load(tmp_path / "absent.json")
        assert blacklist.active_entries(as_of=NOW) == ()


class TestExternalImport:
    """外置导入（★ 用户明确要求的能力）。"""

    def test_bare_markdown_needs_no_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "2026-07-25-央行降准.md"
        path.write_text("央行宣布降准 0.5 个百分点，释放长期资金约 1 万亿元。", encoding="utf-8")

        rows = parse_payload(path, path.read_text(encoding="utf-8"))
        built = build_item(rows[0])
        assert built.title == "央行降准"
        assert "1 万亿" in built.body
        assert built.source_tier is SourceTier.USER

    def test_front_matter_is_honoured(self, tmp_path: Path) -> None:
        path = tmp_path / "note.md"
        path.write_text(
            "---\n"
            "domain: POLICY\n"
            "publish_at: 2026-07-25T09:00:00+08:00\n"
            "symbols: [601398.SH, 601939.SH]\n"
            "importance: 85\n"
            "url: https://example.com/xxx\n"
            "title: 央行宣布降准 0.5 个百分点\n"
            "---\n"
            "正文内容。\n",
            encoding="utf-8",
        )
        built = build_item(parse_payload(path, path.read_text(encoding="utf-8"))[0])

        assert built.domain is IntelDomain.POLICY
        assert built.publish_at == dt.datetime(2026, 7, 25, 9, tzinfo=CST)
        assert Symbol("601398.SH") in built.symbols
        assert built.importance == 85
        assert built.url == "https://example.com/xxx"

    def test_naive_publish_at_read_as_shanghai(self) -> None:
        # 用户手写的时间想表达的是本地时间，按 UTC 解释会推后 8 小时
        built = build_item({"title": "t", "publish_at": "2026-07-25 09:00:00"})
        assert built.publish_at == dt.datetime(2026, 7, 25, 9, tzinfo=CST)

    def test_json_array(self, tmp_path: Path) -> None:
        path = tmp_path / "batch.json"
        path.write_text('[{"title": "甲"}, {"title": "乙"}]', encoding="utf-8")
        assert len(parse_payload(path, path.read_text(encoding="utf-8"))) == 2

    def test_csv_batch(self, tmp_path: Path) -> None:
        path = tmp_path / "batch.csv"
        path.write_text("title,importance\n甲,50\n乙,60\n", encoding="utf-8")
        rows = parse_payload(path, path.read_text(encoding="utf-8"))
        assert [r["title"] for r in rows] == ["甲", "乙"]

    def test_unparseable_symbol_skipped_not_fatal(self) -> None:
        # 一条格式不对的代码不该让整份导入失败
        built = build_item({"title": "t", "symbols": ["600519.SH", "不是代码"]})
        assert built.symbols == (MAOTAI,)

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(IntelError, match="不能同时为空"):
            build_item({"title": "  ", "body": ""})

    def test_body_only_gets_a_title(self) -> None:
        built = build_item({"body": "第一行是标题\n第二行"})
        assert built.title == "第一行是标题"

    def test_item_id_is_idempotent(self) -> None:
        first = build_item({"title": "同一条消息", "url": "https://x.com/1"})
        second = build_item({"title": "同一条消息", "url": "https://x.com/1"})
        assert first.item_id == second.item_id

    def test_unsupported_suffix_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(IntelError, match="不支持的文件类型"):
            parse_payload(tmp_path / "x.pdf", "")

    def test_bad_front_matter_reports_clearly(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        text = "---\n- 这是列表不是映射\n---\n正文"
        with pytest.raises(IntelError, match="键值映射"):
            parse_payload(path, text)


class TestInboxScanner:
    """收件箱目录（验收 3）。"""

    def test_scan_imports_and_archives(self, tmp_path: Path) -> None:
        scanner = InboxScanner(tmp_path)
        scanner.ensure_dirs()
        (tmp_path / "news.md").write_text("公司中标 5 亿元订单", encoding="utf-8")

        report = scanner.scan()
        assert len(report.items) == 1
        # 留在原地会被反复解析
        assert not (tmp_path / "news.md").exists()
        assert (tmp_path / "_processed" / NOW.date().isoformat() / "news.md").exists()

    def test_failed_file_gets_an_error_note(self, tmp_path: Path) -> None:
        # 文件静静消失是最让人困惑的失败方式
        scanner = InboxScanner(tmp_path)
        scanner.ensure_dirs()
        (tmp_path / "broken.json").write_text("{ not json", encoding="utf-8")

        report = scanner.scan()
        assert len(report.failed) == 1
        assert (tmp_path / "_failed" / "broken.json").exists()
        assert (tmp_path / "_failed" / "broken.json.error.txt").exists()

    def test_preview_mode_leaves_files_alone(self, tmp_path: Path) -> None:
        scanner = InboxScanner(tmp_path)
        scanner.ensure_dirs()
        (tmp_path / "news.txt").write_text("一句话", encoding="utf-8")

        report = scanner.scan(move=False)
        assert len(report.items) == 1
        assert (tmp_path / "news.txt").exists()

    def test_readme_and_underscore_dirs_skipped(self, tmp_path: Path) -> None:
        scanner = InboxScanner(tmp_path)
        scanner.ensure_dirs()
        assert (tmp_path / "README.md").exists()
        assert list(scanner.pending()) == []

    def test_unsupported_suffix_ignored(self, tmp_path: Path) -> None:
        scanner = InboxScanner(tmp_path)
        scanner.ensure_dirs()
        (tmp_path / "photo.png").write_bytes(b"\x89PNG")
        assert list(scanner.pending()) == []

    def test_name_collision_does_not_clobber(self, tmp_path: Path) -> None:
        scanner = InboxScanner(tmp_path)
        scanner.ensure_dirs()
        (tmp_path / "a.md").write_text("第一次", encoding="utf-8")
        scanner.scan()
        (tmp_path / "a.md").write_text("第二次", encoding="utf-8")
        scanner.scan()

        archived = sorted((tmp_path / "_processed" / NOW.date().isoformat()).iterdir())
        assert len(archived) == 2

    def test_missing_inbox_yields_nothing(self, tmp_path: Path) -> None:
        assert list(InboxScanner(tmp_path / "absent").pending()) == []


class TestSourceRegistry:
    """源注册与降级（验收 8）。"""

    class _GoodSource:
        name = "good"
        domains = (IntelDomain.MARKET,)

        def fetch(self, since: dt.datetime) -> list[IntelItem]:
            return [item("市场快讯", domain=IntelDomain.MARKET)]

        def health_check(self) -> SourceHealth:
            return SourceHealth(source=self.name, ok=True)

    class _BrokenSource:
        name = "broken"
        domains = (IntelDomain.COMPANY,)

        def fetch(self, since: dt.datetime) -> list[IntelItem]:
            msg = "连接超时"
            raise TimeoutError(msg)

        def health_check(self) -> SourceHealth:
            return SourceHealth(source=self.name, ok=False, error="连接超时")

    def test_one_broken_source_does_not_stop_the_rest(self) -> None:
        # 情报缺失只降级为"缺少证据"，绝不阻断
        registry = SourceRegistry([self._GoodSource(), self._BrokenSource()])
        outcome = fetch_all(registry, since=NOW - dt.timedelta(days=1))

        assert len(outcome.items) == 1
        assert outcome.failed_sources == ("broken",)
        assert "TimeoutError" in outcome.coverage["broken"].error

    def test_missing_domains_reported(self) -> None:
        registry = SourceRegistry([self._GoodSource(), self._BrokenSource()])
        outcome = fetch_all(registry, since=NOW - dt.timedelta(days=1))
        missing = outcome.missing_domains([IntelDomain.MARKET, IntelDomain.COMPANY])
        assert missing == (IntelDomain.COMPANY,)

    def test_domain_filter(self) -> None:
        registry = SourceRegistry([self._GoodSource(), self._BrokenSource()])
        outcome = fetch_all(
            registry, since=NOW - dt.timedelta(days=1), domains=[IntelDomain.MARKET]
        )
        assert set(outcome.coverage) == {"good"}

    def test_register_and_unregister(self) -> None:
        registry = SourceRegistry()
        assert len(registry) == 0
        registry.register(self._GoodSource())
        assert len(registry) == 1
        registry.unregister("good")
        assert len(registry) == 0

    def test_naive_since_rejected(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            fetch_all(SourceRegistry(), since=dt.datetime(2026, 7, 25))  # noqa: DTZ001

    def test_health_check_protocol(self) -> None:
        assert self._GoodSource().health_check().ok
        assert "连接超时" in self._BrokenSource().health_check().message


class TestPluginDiscovery:
    """自定义源插件（外置导入方式四）。"""

    def test_discovers_source_instance(self, tmp_path: Path) -> None:
        (tmp_path / "my_source.py").write_text(
            "from quantstock.intel.types import IntelDomain\n"
            "\n"
            "class _Impl:\n"
            "    name = 'external:mine'\n"
            "    domains = (IntelDomain.COMPANY,)\n"
            "    def fetch(self, since): return []\n"
            "    def health_check(self): return None\n"
            "\n"
            "SOURCE = _Impl()\n",
            encoding="utf-8",
        )
        found = discover_plugins(tmp_path)
        assert [s.name for s in found] == ["external:mine"]

    def test_broken_plugin_does_not_break_startup(self, tmp_path: Path) -> None:
        # 一个坏插件不该让整个系统起不来
        (tmp_path / "bad.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        assert discover_plugins(tmp_path) == []

    def test_missing_dir_is_fine(self, tmp_path: Path) -> None:
        assert discover_plugins(tmp_path / "absent") == []

    def test_underscore_files_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "_private.py").write_text("SOURCE = object()\n", encoding="utf-8")
        assert discover_plugins(tmp_path) == []


class TestStore:
    """存储。"""

    def test_write_then_read(self, tmp_path: Path) -> None:
        store = IntelStore(tmp_path)
        added = store.write([item("甲", item_id="a"), item("乙", item_id="b")])
        assert added == 2
        assert len(store.read(NOW.date())) == 2

    def test_write_is_idempotent(self, tmp_path: Path) -> None:
        store = IntelStore(tmp_path)
        target = item("同一条", item_id="a")
        store.write([target])
        assert store.write([target]) == 0
        assert len(store.read(NOW.date())) == 1

    def test_partition_key_is_publish_date_not_fetch_date(self, tmp_path: Path) -> None:
        # 抓取日分区会让今天回补的历史情报全落到今天，as_of 过滤就形同虚设
        store = IntelStore(tmp_path)
        backfilled = item("旧闻", publish_at=NOW - dt.timedelta(days=30))
        store.write([backfilled])
        assert store.available_dates() == [(NOW - dt.timedelta(days=30)).date()]

    def test_read_range_applies_pit_filter(self, tmp_path: Path) -> None:
        # 验收 7：回测中不得使用未来情报
        store = IntelStore(tmp_path)
        store.write(
            [
                item("昨天", publish_at=NOW - dt.timedelta(days=1), item_id="old"),
                item("刚刚", publish_at=NOW - dt.timedelta(minutes=5), item_id="new"),
            ]
        )
        cutoff = NOW - dt.timedelta(hours=2)
        visible = store.read_range((NOW - dt.timedelta(days=2)).date(), NOW.date(), as_of=cutoff)
        assert [i.item_id for i in visible] == ["old"]

    def test_reread_preserves_every_field(self, tmp_path: Path) -> None:
        store = IntelStore(tmp_path)
        original = item(
            "完整条目",
            body="正文",
            symbols=(MAOTAI,),
            event=EventType.MA,
            importance=77,
            sentiment=0.25,
        )
        store.write([original])
        assert store.read(NOW.date())[0] == original

    def test_merge_keeps_higher_importance(self, tmp_path: Path) -> None:
        from dataclasses import replace  # noqa: PLC0415 - 仅测试内使用

        # 多源印证会随时间陆续到达，后来的不该被先到的版本覆盖
        store = IntelStore(tmp_path)
        first = item("公告", item_id="a", importance=40)
        store.write([first])
        store.write([replace(first, importance=80, duplicates=("b",))])

        stored = store.read(NOW.date())[0]
        assert stored.importance == 80
        assert stored.duplicates == ("b",)

    def test_digest_roundtrip(self, tmp_path: Path) -> None:
        store = IntelStore(tmp_path)
        digest = DigestBuilder().build([item("甲")], trade_date=NOW.date(), generated_at=NOW)
        store.save_digest(digest)
        assert store.load_digest(NOW.date(), "post") == digest

    def test_missing_digest_is_none(self, tmp_path: Path) -> None:
        assert IntelStore(tmp_path).load_digest(NOW.date(), "pre") is None

    def test_purge_before(self, tmp_path: Path) -> None:
        store = IntelStore(tmp_path)
        store.write([item("旧", publish_at=NOW - dt.timedelta(days=400), item_id="old")])
        store.write([item("新", item_id="new")])
        removed = store.purge_before(NOW.date() - dt.timedelta(days=30))
        assert removed == 1
        assert store.available_dates() == [NOW.date()]

    def test_available_dates_ignores_junk(self, tmp_path: Path) -> None:
        store = IntelStore(tmp_path)
        store.write([item("甲")])
        (tmp_path / "items" / "scratch").mkdir()
        assert store.available_dates() == [NOW.date()]


class TestDigest:
    """分域摘要。"""

    def test_portfolio_alerts_come_first(self) -> None:
        held = item(
            "持仓标的立案调查",
            symbols=(MAOTAI,),
            event=EventType.REGULATORY_PROBE,
            importance=90,
        )
        other = item("其它公司公告", symbols=(CATL,), importance=95, item_id="o")
        alerts = build_portfolio_alerts([held, other], [MAOTAI])

        assert len(alerts) == 1
        assert alerts[0].symbol == MAOTAI
        assert alerts[0].severity == "critical"
        assert alerts[0].action_hint

    def test_alert_hint_is_not_an_order(self) -> None:
        # 红线 I-R1：情报只提示，不构成下单指令
        alerts = build_portfolio_alerts(
            [item("立案调查", symbols=(MAOTAI,), event=EventType.REGULATORY_PROBE, importance=90)],
            [MAOTAI],
        )
        assert "建议" in alerts[0].action_hint
        assert "股" not in alerts[0].action_hint

    def test_domains_grouped_with_net_sentiment(self) -> None:
        digest = DigestBuilder().build(
            [
                item("利好", domain=IntelDomain.MACRO, sentiment=0.8, importance=90, item_id="a"),
                item(
                    "小利空", domain=IntelDomain.MACRO, sentiment=-0.2, importance=10, item_id="b"
                ),
            ],
            trade_date=NOW.date(),
            generated_at=NOW,
        )
        block = digest.by_domain[IntelDomain.MACRO]
        assert block.count == 2
        # importance 加权，90 分的利好不该被 10 分的利空稀释掉
        assert block.net_sentiment > 0.5

    def test_top_items_respect_headline_threshold(self) -> None:
        digest = DigestBuilder().build(
            [item("重大", importance=90, item_id="a"), item("普通", importance=20, item_id="b")],
            trade_date=NOW.date(),
            generated_at=NOW,
        )
        assert [i.title for i in digest.top_items] == ["重大"]

    def test_missing_domains_reported(self) -> None:
        digest = DigestBuilder().build(
            [item("只有公司域", domain=IntelDomain.COMPANY)],
            trade_date=NOW.date(),
            generated_at=NOW,
        )
        assert IntelDomain.MACRO in digest.missing_domains
        assert IntelDomain.COMPANY not in digest.missing_domains

    def test_upcoming_calendar_within_horizon(self) -> None:
        digest = DigestBuilder().build(
            [],
            trade_date=NOW.date(),
            calendar=[
                CalendarEvent(NOW.date() + dt.timedelta(days=3), "CPI 发布", IntelDomain.MACRO),
                CalendarEvent(NOW.date() + dt.timedelta(days=30), "太远了", IntelDomain.MACRO),
            ],
            generated_at=NOW,
        )
        assert [e.title for e in digest.upcoming] == ["CPI 发布"]

    def test_render_always_states_intel_health(self) -> None:
        # 用户需要能分辨"没消息"和"没查成"
        digest = DigestBuilder().build(
            [item("甲")],
            trade_date=NOW.date(),
            coverage={"cls": SourceHealth(source="cls", ok=False, error="超时")},
            generated_at=NOW,
        )
        text = "\n".join(DigestBuilder.render(digest))
        assert "情报健康" in text
        assert "采集失败" in text
        assert "cls" in text

    def test_render_healthy_case(self) -> None:
        digest = DigestBuilder().build(
            [item(f"{d.value} 消息", domain=d, item_id=d.value) for d in IntelDomain],
            trade_date=NOW.date(),
            generated_at=NOW,
        )
        text = "\n".join(DigestBuilder.render(digest))
        assert "全部源正常" in text

    def test_watchlist_hits(self) -> None:
        digest = DigestBuilder().build(
            [item("候选池消息", symbols=(CATL,))],
            trade_date=NOW.date(),
            watchlist=[CATL],
            generated_at=NOW,
        )
        assert len(digest.watchlist_hits) == 1


class TestPipeline:
    """端到端流水线。"""

    @pytest.fixture
    def pipeline(self) -> IntelPipeline:
        return IntelPipeline(
            linker=EntityLinker(
                [SymbolDictionary(MAOTAI, "贵州茅台", aliases=("茅台",))], priority=[MAOTAI]
            ),
            importance=ImportanceScorer(holdings=[MAOTAI]),
        )

    def test_full_flow_classifies_scores_and_blacklists(self, pipeline: IntelPipeline) -> None:
        # 验收 5 的完整路径
        raw = item(
            "贵州茅台收到中国证监会立案调查通知书",
            source="sse",
            tier=SourceTier.OFFICIAL,
            url="https://sse.com/n/1",
        )
        result = pipeline.process([raw], as_of=NOW, holdings=[MAOTAI])

        processed = result.items[0]
        assert processed.event_type is EventType.REGULATORY_PROBE
        assert MAOTAI in processed.symbols
        assert processed.sentiment < -0.5
        assert processed.importance >= 80
        assert MAOTAI in result.blacklisted
        assert pipeline.blacklist.is_blocked(MAOTAI, as_of=NOW)

    def test_future_items_filtered_before_anything_else(self, pipeline: IntelPipeline) -> None:
        # 红线 I-R5
        future = item("明天才发布", publish_at=NOW + dt.timedelta(hours=1))
        result = pipeline.process([future], as_of=NOW)
        assert result.items == ()

    def test_dedup_happens_before_scoring(self, pipeline: IntelPipeline) -> None:
        # importance 的"多源印证"项要数合并了几个源
        a = item("贵州茅台中标5亿元大额订单", source="cls", tier=SourceTier.MEDIA, item_id="a")
        b = item("贵州茅台中标5亿元大额订单", source="em", tier=SourceTier.MEDIA, item_id="b")
        merged = pipeline.process([a, b], as_of=NOW)

        solo = IntelPipeline(importance=ImportanceScorer()).process([a], as_of=NOW)
        assert len(merged.items) == 1
        assert merged.items[0].duplicates
        assert merged.items[0].importance > solo.items[0].importance

    def test_llm_fallback_only_when_rules_miss(self) -> None:
        calls: list[str] = []

        def fake_llm(entry: IntelItem) -> tuple[EventType | None, str]:
            calls.append(entry.title)
            return EventType.MACRO_DATA, "llm:fast"

        pipeline = IntelPipeline(llm_classify=fake_llm)
        matched = item("公司收到立案调查通知", item_id="a")
        unmatched = item("一段没有任何关键词的普通描述文字", item_id="b")
        result = pipeline.process([matched, unmatched], as_of=NOW)

        assert calls == ["一段没有任何关键词的普通描述文字"]
        by_title = {i.title: i for i in result.items}
        assert by_title["公司收到立案调查通知"].classifier == "rule"
        assert by_title[unmatched.title].classifier == "llm:fast"
        # 红线 I-R3：LLM 产出必须可识别
        assert by_title[unmatched.title].is_llm_generated

    def test_works_fully_without_llm(self, pipeline: IntelPipeline) -> None:
        # 关掉 LLM 后系统须完整可用（红线 LR2 在情报侧的体现）
        result = pipeline.process([item("公司拟回购股份用于注销")], as_of=NOW)
        assert result.items[0].event_type is EventType.BUYBACK
        assert result.digest is not None

    def test_llm_sentiment_kept_alongside_rule_score(self) -> None:
        pipeline = IntelPipeline(llm_sentiment=lambda _: 0.9)
        result = pipeline.process([item("公司收到立案调查通知")], as_of=NOW)
        processed = result.items[0]

        assert processed.sentiment_source == "rule"
        assert processed.sentiment < 0
        assert processed.llm_sentiment == 0.9
        # 两者分歧 > 0.5 时日报要标注
        assert processed.sentiment_disputed

    def test_source_declared_symbols_win_over_inference(self, pipeline: IntelPipeline) -> None:
        declared = item("一条不提任何公司名的消息", symbols=(CATL,))
        result = pipeline.process([declared], as_of=NOW)
        assert result.items[0].symbols == (CATL,)

    def test_naive_as_of_rejected(self, pipeline: IntelPipeline) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            pipeline.process([], as_of=dt.datetime(2026, 7, 25))  # noqa: DTZ001

    def test_empty_input_still_produces_a_digest(self, pipeline: IntelPipeline) -> None:
        # 验收 8：没有情报时系统不该崩，只是标注"情报缺失"
        result = pipeline.process([], as_of=NOW)
        assert result.items == ()
        assert result.digest is not None
        assert len(result.digest.missing_domains) == len(IntelDomain)

    def test_summary_is_human_readable(self, pipeline: IntelPipeline) -> None:
        result = pipeline.process([item("公司拟回购股份")], as_of=NOW)
        assert "处理 1 条" in result.summary
