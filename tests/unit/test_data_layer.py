"""数据层测试：复权换算、PIT universe、质量校验。

重点覆盖两个最致命的坑：
- 幸存者偏差（DQ11）——回测收益虚高的头号来源
- 复权口径混用（红线 R4）——数字看起来都"像价格"，事后极难发现
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantstock.data.adjust import (
    assert_adjust,
    check_return_consistency,
    convert,
    hfq_to_none,
    hfq_to_qfq,
    none_to_hfq,
)
from quantstock.data.quality import QualityChecker, Severity
from quantstock.data.types import Bar, Instrument, InstrumentStatus, UniverseMember
from quantstock.data.universe import UniverseRegistry, check_survivorship_bias
from quantstock.infra.clock import CST
from quantstock.infra.errors import AdjustMismatchError, DataQualityError
from quantstock.infra.types import Adjust, AssetType, Board, Exchange, Freq, Symbol

MAOTAI = Symbol("600519.SH")
DELISTED = Symbol("600001.SH")
CATL = Symbol("300750.SZ")

D = dt.date
_DEFAULT_DAY = D(2026, 7, 24)
_DEFAULT_LIST_DATE = D(2010, 1, 1)


def bar(
    symbol: Symbol = MAOTAI,
    *,
    day: dt.date = _DEFAULT_DAY,
    o: str = "100",
    h: str = "110",
    low: str = "95",
    c: str = "105",
    volume: int = 1000,
    pre_close: str = "100",
    adjust: Adjust = Adjust.NONE,
) -> Bar:
    """构造一根 K 线，默认值合法。"""
    return Bar(
        symbol=symbol,
        dt=dt.datetime.combine(day, dt.time(15, 0), tzinfo=CST),
        trade_date=day,
        freq=Freq.D,
        adjust=adjust,
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(low),
        close=Decimal(c),
        volume=volume,
        pre_close=Decimal(pre_close),
    )


class TestBar:
    def test_valid_bar__no_problems(self) -> None:
        assert bar().validate() == []

    def test_high_below_close__dq01(self) -> None:
        assert "DQ01" in bar(h="100", c="105").validate()

    def test_negative_price__dq02(self) -> None:
        assert "DQ02" in bar(low="-1").validate()

    def test_negative_volume__dq02(self) -> None:
        assert "DQ02" in bar(volume=-1).validate()

    def test_change_pct(self) -> None:
        assert bar(c="110", pre_close="100").change_pct == Decimal("0.10")

    def test_change_pct__no_pre_close__zero(self) -> None:
        assert bar(pre_close="0").change_pct == Decimal("0")

    def test_limit_up_detection(self) -> None:
        limited = Bar(
            symbol=MAOTAI,
            dt=dt.datetime(2026, 7, 24, 15, 0, tzinfo=CST),
            trade_date=D(2026, 7, 24),
            freq=Freq.D,
            adjust=Adjust.NONE,
            open=Decimal("110"),
            high=Decimal("110"),
            low=Decimal("110"),
            close=Decimal("110"),
            volume=1000,
            pre_close=Decimal("100"),
            limit_up=Decimal("110"),
        )
        assert limited.is_limit_up
        assert not limited.is_limit_down

    def test_suspended__not_tradable(self) -> None:
        suspended = Bar(
            symbol=MAOTAI,
            dt=dt.datetime(2026, 7, 24, 15, 0, tzinfo=CST),
            trade_date=D(2026, 7, 24),
            freq=Freq.D,
            adjust=Adjust.NONE,
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=0,
            is_suspended=True,
        )
        assert not suspended.is_tradable


class TestAdjust:
    def test_assert_adjust__mismatch__raises(self) -> None:
        with pytest.raises(AdjustMismatchError, match="复权口径不匹配"):
            assert_adjust(Adjust.NONE, Adjust.HFQ, context="因子计算")

    def test_assert_adjust__match__passes(self) -> None:
        assert_adjust(Adjust.HFQ, Adjust.HFQ)

    def test_none_to_hfq_roundtrip(self) -> None:
        factor = Decimal("1.5")
        original = Decimal("100")
        assert hfq_to_none(none_to_hfq(original, factor), factor) == original

    def test_hfq_to_qfq(self) -> None:
        # 最新因子 3.0 → 前复权价为后复权价的三分之一
        assert hfq_to_qfq(Decimal("300"), Decimal("3.0")) == Decimal("100.0000")

    def test_qfq_equals_none_at_latest_date(self) -> None:
        """前复权的定义：最新一天的前复权价必须等于不复权真实价。

        这是最容易写错的一处——分母必须是**最新**因子，不是当日因子。
        """
        factor = Decimal("2.5")
        none_price = Decimal("100")
        hfq = none_to_hfq(none_price, factor)
        assert hfq_to_qfq(hfq, factor) == none_price

    def test_qfq_scales_history_down(self) -> None:
        """历史日的前复权价低于当时的真实价——除权把历史整体下移。"""
        none_price = Decimal("100")
        factor_then, factor_latest = Decimal("1.0"), Decimal("2.0")
        hfq = none_to_hfq(none_price, factor_then)
        assert hfq_to_qfq(hfq, factor_latest) < none_price

    @pytest.mark.parametrize("factor", [Decimal("0"), Decimal("-1")])
    def test_invalid_factor__raises(self, factor: Decimal) -> None:
        with pytest.raises(ValueError, match="复权因子必须为正"):
            none_to_hfq(Decimal("100"), factor)

    def test_convert__same_adjust__unchanged(self) -> None:
        assert convert(
            Decimal("100"), source=Adjust.HFQ, target=Adjust.HFQ, adj_factor=Decimal("2")
        ) == Decimal("100")

    def test_convert__qfq_without_latest_factor__raises(self) -> None:
        with pytest.raises(ValueError, match="需要提供 latest_factor"):
            convert(
                Decimal("100"),
                source=Adjust.HFQ,
                target=Adjust.QFQ,
                adj_factor=Decimal("2"),
            )

    def test_convert__none_to_qfq_roundtrip(self) -> None:
        result = convert(
            Decimal("100"),
            source=Adjust.NONE,
            target=Adjust.QFQ,
            adj_factor=Decimal("2"),
            latest_factor=Decimal("4"),
        )
        back = convert(
            result,
            source=Adjust.QFQ,
            target=Adjust.NONE,
            adj_factor=Decimal("2"),
            latest_factor=Decimal("4"),
        )
        assert back == Decimal("100.0000")

    @given(
        price_cents=st.integers(min_value=10, max_value=300_000),
        factor_milli=st.integers(min_value=100, max_value=200_000),
    )
    def test_roundtrip_invariant(self, price_cents: int, factor_milli: int) -> None:
        """不变量：none → hfq → none 回到原值（在量化精度内）。

        取值域刻意限定在真实范围：A 股价格 0.1~3000 元，累计复权因子 0.1~200。
        中间的 hfq 量化到 4 位小数，绝对误差 ≤ 0.00005，回算后被放大 1/factor 倍；
        因子下界 0.1 时最大误差 0.0005，仍在容差内。取值超出该域时（例如 1 分钱
        的股票配上 0.001 的因子）中间值会被压到精度以下，不变量不再成立——
        那种组合在真实市场不存在，不值得为它牺牲存储精度。
        """
        price = Decimal(price_cents) / 100
        factor = Decimal(factor_milli) / 1000
        back = hfq_to_none(none_to_hfq(price, factor), factor)
        assert abs(back - price) <= Decimal("0.001")


class TestReturnConsistency:
    """DQ07：非除权日两种口径的收益率必须一致。"""

    def test_consistent_series__no_mismatch(self) -> None:
        none_prices = [Decimal("100"), Decimal("110"), Decimal("121")]
        hfq_prices = [p * 2 for p in none_prices]  # 等比缩放不改变收益率
        assert check_return_consistency(hfq_prices, none_prices) == []

    def test_inconsistent_series__reports_index(self) -> None:
        none_prices = [Decimal("100"), Decimal("110")]
        hfq_prices = [Decimal("200"), Decimal("240")]  # +20% vs +10%
        assert check_return_consistency(hfq_prices, none_prices) == [1]

    def test_length_mismatch__raises(self) -> None:
        with pytest.raises(ValueError, match="序列长度不一致"):
            check_return_consistency([Decimal("1")], [Decimal("1"), Decimal("2")])


def _instrument(
    symbol: Symbol,
    *,
    list_date: dt.date = _DEFAULT_LIST_DATE,
    delist_date: dt.date | None = None,
) -> Instrument:
    return Instrument(
        symbol=symbol,
        name=symbol,
        asset_type=AssetType.STOCK,
        exchange=Exchange.SH,
        board=Board.MAIN,
        list_date=list_date,
        delist_date=delist_date,
    )


class TestPitUniverse:
    @pytest.fixture
    def registry(self) -> UniverseRegistry:
        return UniverseRegistry(
            [
                _instrument(MAOTAI),
                _instrument(DELISTED, delist_date=D(2023, 6, 1)),
                _instrument(CATL, list_date=D(2018, 6, 11)),
            ],
            members=[
                UniverseMember("hs300", MAOTAI, in_date=D(2010, 1, 1)),
                UniverseMember("hs300", DELISTED, in_date=D(2010, 1, 1), out_date=D(2023, 6, 1)),
            ],
            statuses=[
                InstrumentStatus(DELISTED, start_date=D(2022, 5, 1), is_st=True),
                InstrumentStatus(MAOTAI, start_date=D(2026, 7, 20), is_suspended=True),
            ],
        )

    def test_delisted_instrument_still_present(self, registry: UniverseRegistry) -> None:
        """退市标的必须永久保留——删掉就造成幸存者偏差。"""
        assert registry.instrument(DELISTED) is not None

    def test_listed_symbols__historical_date_includes_later_delisted(
        self, registry: UniverseRegistry
    ) -> None:
        symbols = registry.listed_symbols(D(2022, 1, 1))
        assert DELISTED in symbols

    def test_listed_symbols__after_delisting_excludes_it(self, registry: UniverseRegistry) -> None:
        assert DELISTED not in registry.listed_symbols(D(2024, 1, 1))

    def test_listed_symbols__before_ipo_excludes(self, registry: UniverseRegistry) -> None:
        assert CATL not in registry.listed_symbols(D(2015, 1, 1))
        assert CATL in registry.listed_symbols(D(2019, 1, 1))

    def test_members__respects_out_date(self, registry: UniverseRegistry) -> None:
        assert DELISTED in registry.members("hs300", as_of=D(2022, 1, 1))
        assert DELISTED not in registry.members("hs300", as_of=D(2024, 1, 1))

    def test_is_st__is_time_dependent(self, registry: UniverseRegistry) -> None:
        """ST 状态随时间变化，必须按历史区间查询。"""
        assert not registry.is_st(DELISTED, as_of=D(2021, 1, 1))
        assert registry.is_st(DELISTED, as_of=D(2022, 6, 1))

    def test_is_tradable__suspended__false(self, registry: UniverseRegistry) -> None:
        assert registry.is_tradable(MAOTAI, as_of=D(2026, 7, 1))
        assert not registry.is_tradable(MAOTAI, as_of=D(2026, 7, 24))

    def test_is_tradable__unknown_symbol__false(self, registry: UniverseRegistry) -> None:
        assert not registry.is_tradable(Symbol("999999.SH"), as_of=D(2026, 7, 1))

    def test_filter_tradable(self, registry: UniverseRegistry) -> None:
        result = registry.filter_tradable([MAOTAI, CATL], as_of=D(2026, 7, 24))
        assert result == (CATL,)


class TestSurvivorshipDetection:
    """DQ11：带幸存者偏差的回测结果毫无意义，必须在启动前拦下。"""

    def test_healthy_dataset__passes(self) -> None:
        registry = UniverseRegistry(
            [_instrument(MAOTAI), _instrument(DELISTED, delist_date=D(2023, 6, 1))]
        )
        reports = check_survivorship_bias(
            registry, sample_dates=[D(2022, 1, 1)], today=D(2026, 7, 24)
        )
        assert reports[0].passed
        assert reports[0].delisted_later == 1

    def test_survivors_only__detected_and_raises(self) -> None:
        """只保留当前在市标的的数据湖必须被识别出来。"""
        registry = UniverseRegistry([_instrument(MAOTAI), _instrument(CATL)])
        with pytest.raises(DataQualityError, match="DQ11 幸存者偏差检测未通过"):
            check_survivorship_bias(registry, sample_dates=[D(2022, 1, 1)], today=D(2026, 7, 24))

    def test_survivors_only__report_explains_why(self) -> None:
        registry = UniverseRegistry([_instrument(MAOTAI)])
        reports = check_survivorship_bias(
            registry,
            sample_dates=[D(2022, 1, 1)],
            today=D(2026, 7, 24),
            raise_on_fail=False,
        )
        assert not reports[0].passed
        assert "幸存者偏差" in reports[0].message

    def test_empty_universe__does_not_false_alarm(self) -> None:
        """尚无数据时不应误报——那是"还没导入"而不是"有偏差"。"""
        registry = UniverseRegistry([_instrument(CATL, list_date=D(2018, 6, 11))])
        reports = check_survivorship_bias(
            registry,
            sample_dates=[D(2010, 1, 1)],
            today=D(2026, 7, 24),
            raise_on_fail=False,
        )
        assert reports[0].passed


class TestQualityChecker:
    @pytest.fixture
    def checker(self) -> QualityChecker:
        return QualityChecker()

    @pytest.fixture
    def now(self) -> dt.datetime:
        return dt.datetime(2026, 7, 24, 15, 30, tzinfo=CST)

    def test_clean_batch__passes(self, checker: QualityChecker, now: dt.datetime) -> None:
        report = checker.check_bars([bar(), bar(CATL)], checked_at=now)
        assert report.passed

    def test_dq01_violation__fatal(self, checker: QualityChecker, now: dt.datetime) -> None:
        report = checker.check_bars([bar(h="100", c="105")], checked_at=now)
        assert not report.passed
        assert report.fatal_failures[0].rule == "DQ01"

    def test_dq02_negative_price__fatal(self, checker: QualityChecker, now: dt.datetime) -> None:
        report = checker.check_bars([bar(low="-5")], checked_at=now)
        assert any(r.rule == "DQ02" for r in report.fatal_failures)

    def test_dq03_duplicate_key__fatal(self, checker: QualityChecker, now: dt.datetime) -> None:
        """重复行会让后续 join 静默膨胀，是最难排查的一类问题。"""
        report = checker.check_bars([bar(), bar()], checked_at=now)
        assert any(r.rule == "DQ03" for r in report.fatal_failures)

    def test_raise_if_fatal(self, checker: QualityChecker, now: dt.datetime) -> None:
        report = checker.check_bars([bar(h="1")], checked_at=now)
        with pytest.raises(DataQualityError, match="拒绝入库"):
            report.raise_if_fatal()

    def test_dq05_coverage_below_threshold(self, checker: QualityChecker, now: dt.datetime) -> None:
        expected = [Symbol(f"60000{i}.SH") for i in range(10)]
        report = checker.check_bars([bar(expected[0])], checked_at=now, expected_symbols=expected)
        dq05 = next(r for r in report.results if r.rule == "DQ05")
        assert not dq05.passed
        assert dq05.severity is Severity.SERIOUS

    def test_dq05_full_coverage__passes(self, checker: QualityChecker, now: dt.datetime) -> None:
        report = checker.check_bars(
            [bar(MAOTAI), bar(CATL)], checked_at=now, expected_symbols=[MAOTAI, CATL]
        )
        assert next(r for r in report.results if r.rule == "DQ05").passed

    def test_dq06_price_limit_exceeded(self, checker: QualityChecker, now: dt.datetime) -> None:
        """涨跌幅超限通常意味着除权未被正确处理。"""
        report = checker.check_bars(
            [bar(c="150", pre_close="100", h="150")],
            checked_at=now,
            price_limits={MAOTAI: Decimal("0.10")},
        )
        assert not next(r for r in report.results if r.rule == "DQ06").passed

    def test_dq06_within_limit__passes(self, checker: QualityChecker, now: dt.datetime) -> None:
        report = checker.check_bars(
            [bar(c="105", pre_close="100")],
            checked_at=now,
            price_limits={MAOTAI: Decimal("0.10")},
        )
        assert next(r for r in report.results if r.rule == "DQ06").passed

    def test_dq04_missing_trading_day(self, checker: QualityChecker) -> None:
        result = checker.check_calendar_continuity(
            actual_dates=[D(2026, 7, 24)],
            expected_dates=[D(2026, 7, 23), D(2026, 7, 24)],
        )
        assert not result.passed
        assert result.severity is Severity.FATAL

    def test_dq08_cross_source_deviation(self, checker: QualityChecker) -> None:
        result = checker.check_cross_source(
            primary={MAOTAI: Decimal("100")}, secondary={MAOTAI: Decimal("105")}
        )
        assert not result.passed

    def test_dq08_within_tolerance__passes(self, checker: QualityChecker) -> None:
        result = checker.check_cross_source(
            primary={MAOTAI: Decimal("100.1")}, secondary={MAOTAI: Decimal("100")}
        )
        assert result.passed

    def test_dq09_announcement_before_period_end(self, checker: QualityChecker) -> None:
        """公告日早于报告期结束 → PIT 口径出错，会让回测提前看到数据。"""
        result = checker.check_announcement_dates([(MAOTAI, D(2026, 3, 31), D(2026, 3, 1))])
        assert not result.passed

    def test_dq09_valid__passes(self, checker: QualityChecker) -> None:
        result = checker.check_announcement_dates([(MAOTAI, D(2026, 3, 31), D(2026, 4, 25))])
        assert result.passed

    def test_bad_symbols_collected(self, checker: QualityChecker, now: dt.datetime) -> None:
        report = checker.check_bars([bar(h="1"), bar(CATL)], checked_at=now)
        assert MAOTAI in report.bad_symbols
