"""主干链路测试：数据 → 建议 → 回测。

**这是之前最该有却没有的一类测试。** 1047 个测试全绿的时候，
用户依然拿不到一条建议——因为每个测试都只在测自己那一层，
没有一个测试问过"从数据到建议这条路走得通吗"。

所以这里的断言都是端到端的：喂进 CSV 行情，断言真的产出了带四支柱的计划。
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import random
from decimal import Decimal
from pathlib import Path

import pytest

from quantstock.config.models import RootConfig
from quantstock.config.settings import Secrets, Settings
from quantstock.data.sources.csv_source import CsvSource
from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import DataError, DataSourceError
from quantstock.infra.money import money
from quantstock.infra.types import Adjust, AssetType, Board, Symbol
from quantstock.services.advisor_service import AdvisorService
from quantstock.services.backtest_service import BacktestService
from quantstock.services.data_service import CORE_UNIVERSE, DataService, build_source

NOW = dt.datetime(2026, 7, 24, 16, 0, tzinfo=CST)
TODAY = NOW.date()

SPECS: dict[str, tuple[float, float]] = {
    "510300.SH": (4.20, 0.00045),
    "510500.SH": (6.10, 0.00030),
    "588000.SH": (1.05, -0.00035),
    "159915.SZ": (2.30, 0.00015),
    "511990.SH": (100.0, 0.00006),
    "600519.SH": (1580.0, 0.00050),
    "601318.SH": (48.0, -0.00020),
    "000858.SZ": (140.0, 0.00025),
    "300750.SZ": (205.0, 0.00040),
    "002415.SZ": (31.0, -0.00010),
}


@pytest.fixture(autouse=True)
def _frozen() -> None:
    """固定时钟。"""
    set_clock(FrozenClock(NOW))


def write_market_data(root: Path, *, days: int = 300, seed: int = 42) -> list[dt.date]:
    """生成合成行情 CSV。

    刻意用**随机游走**而不是精心构造的上涨序列：动量策略在纯噪声上本来就
    不该赚钱，若测试里它赚了，说明链路某处泄漏了未来信息。

    Args:
        root: CSV 根目录。
        days: 交易日数。
        seed: 随机种子。

    Returns:
        交易日列表。
    """
    bars = root / "bars"
    bars.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    dates: list[dt.date] = []
    cursor = TODAY
    while len(dates) < days:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor -= dt.timedelta(days=1)
    dates.reverse()

    for symbol, (start_price, drift) in SPECS.items():
        rows: list[list[object]] = []
        price = start_price
        for day in dates:
            price *= math.exp(drift + rng.gauss(0, 0.012))
            open_ = price * (1 + rng.gauss(0, 0.003))
            high = max(open_, price) * (1 + abs(rng.gauss(0, 0.004)))
            low = min(open_, price) * (1 - abs(rng.gauss(0, 0.004)))
            rows.append(
                [
                    day.isoformat(),
                    f"{open_:.3f}",
                    f"{high:.3f}",
                    f"{low:.3f}",
                    f"{price:.3f}",
                    int(rng.uniform(2e6, 8e6)),
                    f"{price * rng.uniform(2e6, 8e6):.0f}",
                    rows[-1][4] if rows else f"{price:.3f}",
                ]
            )
        with (bars / f"{symbol}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["date", "open", "high", "low", "close", "volume", "amount", "pre_close"]
            )
            writer.writerows(rows)
    return dates


def make_settings(tmp_path: Path) -> Settings:
    """构造指向临时目录的配置。

    Args:
        tmp_path: 临时目录。

    Returns:
        Settings 实例。
    """
    config = RootConfig.model_validate(
        {
            "app": {"var_dir": str(tmp_path / "var")},
            "data": {"source_chain": ["csv"], "start_date": "2024-01-01"},
            "execution": {"broker": "manual"},
        }
    )
    return Settings(config=config, secrets=Secrets(), config_dir=tmp_path / "config")


@pytest.fixture
def ready(tmp_path: Path) -> tuple[Settings, DataService]:
    """准备好数据湖的环境。

    Args:
        tmp_path: 临时目录。

    Returns:
        ``(配置, 已初始化的数据服务)``。
    """
    settings = make_settings(tmp_path)
    write_market_data(settings.var_dir / "csv")
    service = DataService(settings, source=CsvSource(settings.var_dir / "csv"))
    service.sync_instruments()
    service.update(CORE_UNIVERSE, start=dt.date(2024, 1, 1))
    return settings, service


class TestCsvSource:
    """CSV 数据源。"""

    def test_reads_bars(self, tmp_path: Path) -> None:
        write_market_data(tmp_path, days=30)
        source = CsvSource(tmp_path)
        bars = source.fetch_daily_bars(
            [Symbol("510300.SH")],
            start=TODAY - dt.timedelta(days=60),
            end=TODAY,
            adjust=Adjust.HFQ,
        )
        assert bars
        assert all(b.adjust is Adjust.HFQ for b in bars)
        assert bars == sorted(bars, key=lambda b: b.trade_date)

    def test_chinese_headers_accepted(self, tmp_path: Path) -> None:
        # 各家软件导出的表头五花八门，为此让用户手工改列名是没必要的摩擦
        (tmp_path / "bars").mkdir(parents=True)
        (tmp_path / "bars" / "600519.SH.csv").write_text(
            "日期,开盘,最高,最低,收盘,成交量\n2026-07-24,1580,1600,1570,1590,1000\n",
            encoding="utf-8",
        )
        bars = CsvSource(tmp_path).fetch_daily_bars(
            [Symbol("600519.SH")], start=TODAY, end=TODAY, adjust=Adjust.NONE
        )
        assert len(bars) == 1
        assert bars[0].close == Decimal("1590")

    @pytest.mark.parametrize("raw", ["2026-07-24", "20260724", "2026/07/24"])
    def test_date_formats(self, tmp_path: Path, raw: str) -> None:
        (tmp_path / "bars").mkdir(parents=True)
        (tmp_path / "bars" / "600519.SH.csv").write_text(
            f"date,close\n{raw},1590\n", encoding="utf-8"
        )
        bars = CsvSource(tmp_path).fetch_daily_bars(
            [Symbol("600519.SH")], start=TODAY, end=TODAY, adjust=Adjust.NONE
        )
        assert bars[0].trade_date == dt.date(2026, 7, 24)

    def test_prices_never_pass_through_float(self, tmp_path: Path) -> None:
        # CSV 里的价格是精确的十进制文本，经过 float 就再也回不去了（红线 R1）
        (tmp_path / "bars").mkdir(parents=True)
        (tmp_path / "bars" / "600519.SH.csv").write_text(
            "date,close\n2026-07-24,0.1\n", encoding="utf-8"
        )
        bars = CsvSource(tmp_path).fetch_daily_bars(
            [Symbol("600519.SH")], start=TODAY, end=TODAY, adjust=Adjust.NONE
        )
        assert bars[0].close == Decimal("0.1")
        assert str(bars[0].close) == "0.1"

    def test_missing_file_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        # 批量拉取时一只缺失不该让整批失败
        write_market_data(tmp_path, days=5)
        bars = CsvSource(tmp_path).fetch_daily_bars(
            [Symbol("510300.SH"), Symbol("999999.SH")],
            start=TODAY - dt.timedelta(days=30),
            end=TODAY,
            adjust=Adjust.NONE,
        )
        assert {b.symbol for b in bars} == {Symbol("510300.SH")}

    def test_missing_close_column_raises(self, tmp_path: Path) -> None:
        (tmp_path / "bars").mkdir(parents=True)
        (tmp_path / "bars" / "600519.SH.csv").write_text(
            "date,open\n2026-07-24,1\n", encoding="utf-8"
        )
        with pytest.raises(DataSourceError, match="缺少日期或收盘价"):
            CsvSource(tmp_path).fetch_daily_bars(
                [Symbol("600519.SH")], start=TODAY, end=TODAY, adjust=Adjust.NONE
            )

    def test_instruments_inferred_from_bars(self, tmp_path: Path) -> None:
        # 用户只丢了 bars 目录进来也应该能跑
        write_market_data(tmp_path, days=5)
        instruments = CsvSource(tmp_path).fetch_instruments()
        assert len(instruments) == len(SPECS)
        by_symbol = {i.symbol: i for i in instruments}
        assert by_symbol[Symbol("510300.SH")].asset_type is AssetType.ETF
        assert by_symbol[Symbol("300750.SZ")].board is Board.GEM
        # 588000 跟踪科创50，但它本身是一只 ETF——按 STAR 处理会让它错误地
        # 继承科创板的 200 股起与 ±20% 涨跌幅
        assert by_symbol[Symbol("588000.SH")].board is Board.ETF
        assert by_symbol[Symbol("600519.SH")].board is Board.MAIN

    def test_trading_days_inferred_from_bars(self, tmp_path: Path) -> None:
        # 比"假设周一到周五都是交易日"准确得多——后者会把春节长假算进去
        dates = write_market_data(tmp_path, days=20)
        days = CsvSource(tmp_path).fetch_trading_days(start=dates[0], end=dates[-1])
        assert days == dates

    def test_health_check(self, tmp_path: Path) -> None:
        assert not CsvSource(tmp_path / "absent").health_check().ok
        write_market_data(tmp_path, days=5)
        assert CsvSource(tmp_path).health_check().ok

    def test_write_bars_roundtrip(self, tmp_path: Path) -> None:
        write_market_data(tmp_path, days=10)
        source = CsvSource(tmp_path)
        bars = source.fetch_daily_bars(
            [Symbol("510300.SH")],
            start=TODAY - dt.timedelta(days=30),
            end=TODAY,
            adjust=Adjust.NONE,
        )
        exported = CsvSource(tmp_path / "out")
        assert exported.write_bars(bars) == len(bars)
        assert len(
            exported.fetch_daily_bars(
                [Symbol("510300.SH")],
                start=TODAY - dt.timedelta(days=30),
                end=TODAY,
                adjust=Adjust.NONE,
            )
        ) == len(bars)


class TestDataService:
    """数据服务。"""

    def test_init_writes_lake(self, ready: tuple[Settings, DataService]) -> None:
        _, service = ready
        status = service.status()
        assert status.is_ready
        assert status.symbols == len(SPECS)
        assert status.latest_date == TODAY

    def test_empty_lake_reports_not_ready(self, tmp_path: Path) -> None:
        status = DataService(make_settings(tmp_path)).status()
        assert not status.is_ready
        assert "data init" in status.message

    def test_incremental_update_does_not_refetch_everything(
        self, ready: tuple[Settings, DataService]
    ) -> None:
        # 每次都从头重拉是最容易被忽略的性能坑
        _, service = ready
        report = service.update(CORE_UNIVERSE)
        assert report.start > dt.date(2024, 1, 1)

    def test_read_bars_returns_sorted(self, ready: tuple[Settings, DataService]) -> None:
        _, service = ready
        history = service.read_bars(
            [Symbol("510300.SH")], start=TODAY - dt.timedelta(days=90), end=TODAY
        )
        bars = history[Symbol("510300.SH")]
        assert bars == sorted(bars, key=lambda b: b.trade_date)

    def test_unknown_source_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="未知的数据源"):
            build_source(make_settings(tmp_path), kind="nope")

    def test_fallback_chain_satisfies_the_source_protocol(self, tmp_path: Path) -> None:
        # 降级链必须能当成普通数据源即插即用，否则上层就得到处写 isinstance 分支
        from quantstock.data.protocols import FallbackChain, MarketDataSource  # noqa: PLC0415

        chain = FallbackChain([CsvSource(tmp_path), CsvSource(tmp_path / "b")])
        assert isinstance(chain, MarketDataSource)
        assert "chain(" in chain.name
        assert chain.health_check() is not None


class TestAdviseChain:
    """建议链路——之前完全断裂的那一段。"""

    def test_produces_a_plan_from_raw_data(self, ready: tuple[Settings, DataService]) -> None:
        settings, data = ready
        result = AdvisorService(settings, data=data).advise(
            as_of=TODAY, total_value=money("200000")
        )
        assert result.plan.intents, "应该产出至少一条建议"
        assert result.plan.data_fingerprint.startswith("sha256:")
        assert result.plan.param_hash

    def test_every_intent_carries_four_pillars(self, ready: tuple[Settings, DataService]) -> None:
        # 宁可不建议，也不给无法解释的建议
        settings, data = ready
        result = AdvisorService(settings, data=data).advise(
            as_of=TODAY, total_value=money("200000")
        )
        for intent in result.plan.intents:
            assert intent.rationale.is_complete, intent.rationale.missing_pillars()
            assert intent.rationale.quant_evidence
            assert intent.rationale.falsification or intent.rationale.counter_evidence

    def test_intents_are_actionable(self, ready: tuple[Settings, DataService]) -> None:
        # "建议减仓"不可执行，"卖出 400 股，限价 1578~1596"才可以
        settings, data = ready
        result = AdvisorService(settings, data=data).advise(
            as_of=TODAY, total_value=money("200000")
        )
        for intent in result.plan.intents:
            assert intent.qty > 0
            assert intent.qty % 100 == 0, "A 股必须整手"
            assert intent.price_low <= intent.price_high

    def test_plan_is_persisted_and_reloadable(self, ready: tuple[Settings, DataService]) -> None:
        settings, data = ready
        service = AdvisorService(settings, data=data)
        result = service.advise(as_of=TODAY, total_value=money("200000"))

        reloaded = service.store.load(TODAY, result.plan.plan_id)
        assert reloaded.plan_id == result.plan.plan_id
        assert len(reloaded.intents) == len(result.plan.intents)

    def test_llm_off_still_produces_advice(self, ready: tuple[Settings, DataService]) -> None:
        # 红线 LR2：关掉 LLM 后系统必须完整可用
        settings, data = ready
        result = AdvisorService(settings, data=data).advise(
            as_of=TODAY, total_value=money("200000")
        )
        assert not result.llm_used
        assert result.final_scores == result.base_scores
        assert "纯量化" in result.summary

    def test_empty_lake_fails_loudly(self, tmp_path: Path) -> None:
        # 行情缺失必须停机——拿不准的价格算不出对的仓位
        settings = make_settings(tmp_path)
        with pytest.raises(DataError, match="data init"):
            AdvisorService(settings).advise(as_of=TODAY)

    def test_short_history_is_rejected(self, tmp_path: Path) -> None:
        # 硬算 MA60 会得到一个用短序列凑出来的假均线
        settings = make_settings(tmp_path)
        write_market_data(settings.var_dir / "csv", days=10)
        data = DataService(settings, source=CsvSource(settings.var_dir / "csv"))
        data.update(CORE_UNIVERSE, start=dt.date(2024, 1, 1))
        with pytest.raises(DataError, match="不足"):
            AdvisorService(settings, data=data).advise(as_of=TODAY)

    def test_pit_no_future_bars(self, ready: tuple[Settings, DataService]) -> None:
        # 红线 R2：拿一个历史日期出建议，指纹里不能出现之后的数据
        settings, data = ready
        past = TODAY - dt.timedelta(days=30)
        service = AdvisorService(settings, data=data)
        first = service.advise(as_of=past, total_value=money("200000"), save=False)
        second = service.advise(as_of=past, total_value=money("200000"), save=False)
        assert first.plan.data_fingerprint == second.plan.data_fingerprint

    def test_intel_blacklist_blocks_buying(self, ready: tuple[Settings, DataService]) -> None:
        # 情报的单向否决必须真的传导到建议里（红线 I-R2）
        from quantstock.services.intel_service import IntelService  # noqa: PLC0415
        from tests.unit.test_intel import item  # noqa: PLC0415

        settings, data = ready
        intel = IntelService(settings)
        blocked = Symbol("600519.SH")
        intel.ingest(
            [
                item(
                    "贵州茅台收到中国证监会立案调查通知书",
                    source="sse",
                    tier=__import__(
                        "quantstock.intel.types", fromlist=["SourceTier"]
                    ).SourceTier.OFFICIAL,
                    symbols=(blocked,),
                    url="https://sse.com.cn/n/1",
                    publish_at=NOW - dt.timedelta(hours=2),
                )
            ]
        )
        assert intel.is_blocked(blocked)

        result = AdvisorService(settings, data=data, intel=intel).advise(
            as_of=TODAY, total_value=money("200000")
        )
        buys = {i.symbol for i in result.plan.intents if i.side.value == "buy"}
        assert blocked not in buys


class TestBacktestChain:
    """回测链路——同样是之前零调用方的模块。"""

    def test_runs_over_a_historical_range(self, ready: tuple[Settings, DataService]) -> None:
        settings, data = ready
        report = BacktestService(settings, data=data).run(
            start=TODAY - dt.timedelta(days=300),
            end=TODAY,
            initial_cash=money("200000"),
        )
        assert report.trading_days > 0
        assert report.stats.trading_days > 0
        assert report.trial_id

    def test_every_run_is_recorded(self, ready: tuple[Settings, DataService]) -> None:
        # 删掉失败的尝试会让 DSR 系统性偏乐观
        settings, data = ready
        service = BacktestService(settings, data=data)
        for _ in range(3):
            service.run(
                start=TODAY - dt.timedelta(days=300), end=TODAY, initial_cash=money("200000")
            )
        assert service.trials.count("daily_advice") == 3

    def test_test_segment_use_is_detectable(self, ready: tuple[Settings, DataService]) -> None:
        # 测试集一次性使用
        settings, data = ready
        service = BacktestService(settings, data=data)
        assert not service.trials.test_segment_used("daily_advice")
        service.run(
            start=TODAY - dt.timedelta(days=300),
            end=TODAY,
            initial_cash=money("200000"),
            segment="test",
        )
        assert service.trials.test_segment_used("daily_advice")

    def test_random_walk_does_not_produce_free_money(
        self, ready: tuple[Settings, DataService]
    ) -> None:
        # 合成数据是随机游走，动量策略在纯噪声上本就不该赚钱。
        # 若这里跑出高收益，说明链路某处泄漏了未来信息
        settings, data = ready
        report = BacktestService(settings, data=data).run(
            start=TODAY - dt.timedelta(days=300),
            end=TODAY,
            initial_cash=money("200000"),
        )
        assert report.stats.sharpe < 2.0, (
            f"随机游走上跑出 Sharpe {report.stats.sharpe:.2f}，检查是否有未来函数"
        )

    def test_report_warns_about_short_window(self, ready: tuple[Settings, DataService]) -> None:
        settings, data = ready
        report = BacktestService(settings, data=data).run(
            start=TODAY - dt.timedelta(days=90),
            end=TODAY,
            initial_cash=money("200000"),
        )
        assert any("不足一年" in w for w in report.warnings())

    def test_empty_range_fails_loudly(self, ready: tuple[Settings, DataService]) -> None:
        settings, data = ready
        with pytest.raises(DataError):
            BacktestService(settings, data=data).run(
                start=dt.date(2020, 1, 1), end=dt.date(2020, 2, 1)
            )

    def test_backtest_advisor_forces_llm_replay(self, ready: tuple[Settings, DataService]) -> None:
        # 红线 LR3：回测里绝不允许实时调用
        settings, data = ready
        advisor = BacktestService(settings, data=data).advisor()
        assert advisor is not None
