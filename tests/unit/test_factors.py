"""因子层测试。

核心用例是 `TestUnbuyableSampleBias`——量化"入场日涨停样本不剔除会让 IC 虚高多少"。
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

from quantstock.factors.pipeline import (
    build_labels,
    compute_ic,
    fill_missing,
    layer_backtest,
    neutralize,
    rank_pct,
    standardize,
    winsorize,
)
from quantstock.factors.technical import (
    atr,
    bias,
    drawdown_from_peak,
    ema,
    macd,
    momentum,
    moving_average,
    position_in_range,
    realized_volatility,
    reversal,
    rsi,
    volume_ratio,
)
from quantstock.factors.types import FactorCategory, FactorMeta, FactorPanel
from quantstock.infra.types import Symbol

A = Symbol("600000.SH")
B = Symbol("600001.SH")
C = Symbol("600002.SH")
DE = Symbol("600003.SH")
E = Symbol("600004.SH")


class TestFactorMeta:
    def test_valid_meta(self) -> None:
        meta = FactorMeta(
            name="momentum_60d",
            category=FactorCategory.TECHNICAL,
            direction=1,
            lookback=60,
            description="60 日动量",
        )
        assert meta.direction == 1

    @pytest.mark.parametrize("direction", [0, 2, -2])
    def test_invalid_direction__raises(self, direction: int) -> None:
        """方向标错会让策略从赚钱变成亏钱，而回测曲线看上去仍然"有信号"。"""
        with pytest.raises(ValueError, match="方向必须是"):
            FactorMeta(
                name="x",
                category=FactorCategory.TECHNICAL,
                direction=direction,
                lookback=1,
                description="",
            )

    def test_negative_lookback__raises(self) -> None:
        with pytest.raises(ValueError, match="回看期不能为负"):
            FactorMeta(
                name="x",
                category=FactorCategory.TECHNICAL,
                direction=1,
                lookback=-1,
                description="",
            )


class TestMovingAverage:
    def test_simple_average(self) -> None:
        assert moving_average([1, 2, 3, 4, 5], 5) == 3.0

    def test_uses_only_recent_window(self) -> None:
        assert moving_average([100, 1, 2, 3], 3) == 2.0

    def test_insufficient_data__raises(self) -> None:
        with pytest.raises(ValueError, match="数据不足"):
            moving_average([1, 2], 5)

    def test_zero_window__raises(self) -> None:
        with pytest.raises(ValueError, match="窗口必须为正"):
            moving_average([1, 2, 3], 0)


class TestEma:
    def test_constant_series__equals_constant(self) -> None:
        assert ema([5.0] * 20, 5) == pytest.approx(5.0)

    def test_reacts_faster_than_sma(self) -> None:
        """上涨行情中 EMA 应高于同周期 SMA——这正是用它做择时的理由。"""
        prices = [float(i) for i in range(1, 21)]
        assert ema(prices, 5) > moving_average(prices, 5)

    def test_empty__raises(self) -> None:
        with pytest.raises(ValueError, match="价格序列为空"):
            ema([], 5)


class TestMomentum:
    def test_positive_momentum(self) -> None:
        assert momentum([100, 110], 1) == pytest.approx(0.10)

    def test_negative_momentum(self) -> None:
        assert momentum([100, 90], 1) == pytest.approx(-0.10)

    def test_skip_recent__excludes_latest_bars(self) -> None:
        """12-1 动量跳过最近一段，避免与短期反转互相抵消。"""
        prices = [100.0, 110.0, 200.0]
        assert momentum(prices, 1, skip_recent=1) == pytest.approx(0.10)

    def test_insufficient_data__raises(self) -> None:
        with pytest.raises(ValueError, match="数据不足"):
            momentum([100.0], 5)

    def test_negative_skip__raises(self) -> None:
        with pytest.raises(ValueError, match="skip_recent 不能为负"):
            momentum([100.0, 110.0], 1, skip_recent=-1)

    def test_reversal_is_negated_momentum(self) -> None:
        prices = [100.0, 90.0]
        assert reversal(prices, 1) == -momentum(prices, 1)


class TestVolatility:
    def test_constant_prices__zero_volatility(self) -> None:
        assert realized_volatility([100.0] * 30, 20) == pytest.approx(0.0)

    def test_annualization_scales_up(self) -> None:
        prices = [100.0 * (1.01**i) if i % 2 else 100.0 * (0.99**i) for i in range(30)]
        raw = realized_volatility(prices, 20, annualize=False)
        annual = realized_volatility(prices, 20, annualize=True)
        assert annual == pytest.approx(raw * math.sqrt(252))


class TestAtr:
    def test_constant_range(self) -> None:
        highs = [105.0] * 25
        lows = [95.0] * 25
        closes = [100.0] * 25
        assert atr(highs, lows, closes, 20) == pytest.approx(10.0)

    def test_mismatched_lengths__raises(self) -> None:
        with pytest.raises(ValueError, match="长度必须一致"):
            atr([1.0, 2.0], [1.0], [1.0, 2.0])

    def test_insufficient_data__raises(self) -> None:
        with pytest.raises(ValueError, match="数据不足"):
            atr([1.0], [1.0], [1.0], 20)


class TestRsi:
    def test_all_gains__hundred(self) -> None:
        assert rsi([float(i) for i in range(1, 20)], 14) == 100.0

    def test_all_losses__near_zero(self) -> None:
        assert rsi([float(i) for i in range(20, 1, -1)], 14) == pytest.approx(0.0)

    def test_balanced__near_fifty(self) -> None:
        prices = [100.0]
        for i in range(14):
            prices.append(prices[-1] + (1 if i % 2 == 0 else -1))
        assert 40 < rsi(prices, 14) < 60


class TestMacd:
    def test_returns_three_values(self) -> None:
        prices = [100.0 + i for i in range(40)]
        dif, dea, hist = macd(prices)
        assert hist == pytest.approx((dif - dea) * 2)

    def test_uptrend__positive_dif(self) -> None:
        assert macd([100.0 + i * 2 for i in range(40)])[0] > 0

    def test_fast_not_less_than_slow__raises(self) -> None:
        with pytest.raises(ValueError, match="快线周期必须小于慢线周期"):
            macd([100.0] * 40, fast=26, slow=12)


class TestOtherIndicators:
    def test_bias(self) -> None:
        assert bias([1.0, 2.0, 3.0], 3) == pytest.approx(0.5)

    def test_bias_zero_ma__raises(self) -> None:
        with pytest.raises(ValueError, match="均线值为零"):
            bias([-1.0, 0.0, 1.0], 3)

    def test_volume_ratio(self) -> None:
        assert volume_ratio([100.0] * 20 + [200.0], 20) == pytest.approx(2.0)

    def test_volume_ratio_zero_baseline__zero(self) -> None:
        assert volume_ratio([0.0] * 20 + [100.0], 20) == 0.0

    @pytest.mark.parametrize(
        ("prices", "expected"),
        [([1.0, 2.0, 3.0], 1.0), ([3.0, 2.0, 1.0], 0.0), ([1.0, 3.0, 2.0], 0.5)],
    )
    def test_position_in_range(self, prices: list[float], expected: float) -> None:
        assert position_in_range(prices, 3) == pytest.approx(expected)

    def test_position_in_range__flat__midpoint(self) -> None:
        assert position_in_range([5.0] * 10, 10) == 0.5

    def test_drawdown_from_peak(self) -> None:
        assert drawdown_from_peak([100.0, 120.0, 90.0]) == pytest.approx(-0.25)

    def test_drawdown_at_peak__zero(self) -> None:
        assert drawdown_from_peak([100.0, 120.0]) == pytest.approx(0.0)

    def test_drawdown_empty__raises(self) -> None:
        with pytest.raises(ValueError, match="价格序列为空"):
            drawdown_from_peak([])


class TestWinsorize:
    def test_clips_outliers(self) -> None:
        values = {A: 1.0, B: 2.0, C: 3.0, DE: 4.0, E: 1000.0}
        result = winsorize(values, n_mad=3.0)
        assert result[E] < 1000.0
        assert result[B] == 2.0

    def test_uniform_values__unchanged(self) -> None:
        values = {A: 5.0, B: 5.0, C: 5.0}
        assert winsorize(values) == values

    def test_empty__empty(self) -> None:
        assert winsorize({}) == {}

    def test_invalid_n_mad__raises(self) -> None:
        with pytest.raises(ValueError, match="n_mad 必须为正"):
            winsorize({A: 1.0}, n_mad=0)


class TestFillMissing:
    def test_uses_industry_median(self) -> None:
        values: dict[Symbol, float | None] = {A: 1.0, B: 3.0, C: None}
        groups = {A: "银行", B: "银行", C: "银行"}
        assert fill_missing(values, groups=groups)[C] == 2.0

    def test_falls_back_to_overall_median(self) -> None:
        values: dict[Symbol, float | None] = {A: 1.0, B: 3.0, C: None}
        assert fill_missing(values)[C] == 2.0

    def test_all_missing__empty(self) -> None:
        assert fill_missing({A: None, B: None}) == {}

    def test_unknown_group__uses_overall(self) -> None:
        values: dict[Symbol, float | None] = {A: 1.0, B: 3.0, C: None}
        groups = {A: "银行", B: "银行", C: "未知行业"}
        assert fill_missing(values, groups=groups)[C] == 2.0


class TestStandardize:
    def test_zero_mean_unit_std(self) -> None:
        result = standardize({A: 1.0, B: 2.0, C: 3.0})
        assert sum(result.values()) == pytest.approx(0.0)
        assert result[C] > result[B] > result[A]

    def test_constant_values__all_zero(self) -> None:
        assert set(standardize({A: 5.0, B: 5.0}).values()) == {0.0}

    def test_single_value__zero(self) -> None:
        assert standardize({A: 5.0}) == {A: 0.0}


class TestRankPct:
    def test_ordering(self) -> None:
        result = rank_pct({A: 10.0, B: 20.0, C: 30.0})
        assert result[A] == 0.0
        assert result[B] == 0.5
        assert result[C] == 1.0

    def test_single__midpoint(self) -> None:
        assert rank_pct({A: 1.0}) == {A: 0.5}

    def test_empty(self) -> None:
        assert rank_pct({}) == {}


class TestNeutralize:
    def test_removes_industry_mean(self) -> None:
        """中性化后，"高动量"不再等同于"处在当期最强的行业"。"""
        values = {A: 10.0, B: 12.0, C: 1.0, DE: 3.0}
        groups = {A: "白酒", B: "白酒", C: "银行", DE: "银行"}
        result = neutralize(values, groups=groups)
        assert result[A] == pytest.approx(-1.0)
        assert result[B] == pytest.approx(1.0)
        assert result[C] == pytest.approx(-1.0)
        assert result[DE] == pytest.approx(1.0)

    def test_empty(self) -> None:
        assert neutralize({}, groups={}) == {}


class TestBuildLabels:
    def test_marks_unbuyable_samples(self) -> None:
        samples = build_labels(
            factor_values={A: 1.0, B: 2.0},
            forward_returns={A: 0.1, B: 0.2},
            unbuyable_at_entry=frozenset({B}),
        )
        by_symbol = {s.symbol: s for s in samples}
        assert by_symbol[A].usable
        assert not by_symbol[B].usable
        assert "无法买入" in by_symbol[B].excluded_reason

    def test_marks_unsellable_samples(self) -> None:
        samples = build_labels(
            factor_values={A: 1.0},
            forward_returns={A: 0.1},
            unsellable_at_exit=frozenset({A}),
        )
        assert "无法卖出" in samples[0].excluded_reason

    def test_can_disable_cleaning_for_comparison(self) -> None:
        samples = build_labels(
            factor_values={A: 1.0},
            forward_returns={A: 0.1},
            unbuyable_at_entry=frozenset({A}),
            exclude_unbuyable=False,
        )
        assert samples[0].usable

    def test_missing_forward_return__skipped(self) -> None:
        samples = build_labels(factor_values={A: 1.0, B: 2.0}, forward_returns={A: 0.1})
        assert [s.symbol for s in samples] == [A]


class TestComputeIC:
    def test_perfect_positive_correlation(self) -> None:
        samples = build_labels(
            factor_values={A: 1.0, B: 2.0, C: 3.0},
            forward_returns={A: 0.01, B: 0.02, C: 0.03},
        )
        assert compute_ic("f", [samples]).ic_mean == pytest.approx(1.0)

    def test_perfect_negative_correlation(self) -> None:
        samples = build_labels(
            factor_values={A: 1.0, B: 2.0, C: 3.0},
            forward_returns={A: 0.03, B: 0.02, C: 0.01},
        )
        assert compute_ic("f", [samples]).ic_mean == pytest.approx(-1.0)

    def test_multi_period_stats(self) -> None:
        periods = [
            build_labels(
                factor_values={A: 1.0, B: 2.0, C: 3.0},
                forward_returns={A: 0.01, B: 0.02, C: 0.03},
            )
            for _ in range(5)
        ]
        stats = compute_ic("f", periods)
        assert stats.periods == 5
        assert stats.positive_rate == 1.0

    def test_no_usable_samples__raises(self) -> None:
        samples = build_labels(
            factor_values={A: 1.0, B: 2.0},
            forward_returns={A: 0.1, B: 0.2},
            unbuyable_at_entry=frozenset({A, B}),
        )
        with pytest.raises(ValueError, match="没有任何一期有足够的可用样本"):
            compute_ic("f", [samples])


class TestUnbuyableSampleBias:
    """量化"入场日涨停样本不剔除"造成的 IC 虚高。

    构造一个只对涨停股"有效"的因子：涨停股（买不到）未来大涨，
    其余股票的因子值与未来收益完全无关。
    """

    def test_ic_collapses_after_cleaning(self) -> None:
        limit_up = frozenset({DE, E})
        factor_values = {A: 1.0, B: 2.0, C: 3.0, DE: 4.0, E: 5.0}
        # 前三只：因子与收益负相关；后两只涨停股：高因子高收益
        forward_returns = {A: 0.03, B: 0.02, C: 0.01, DE: 0.20, E: 0.30}

        dirty = compute_ic(
            "fake_alpha",
            [
                build_labels(
                    factor_values=factor_values,
                    forward_returns=forward_returns,
                    unbuyable_at_entry=limit_up,
                    exclude_unbuyable=False,
                )
            ],
        )
        clean = compute_ic(
            "fake_alpha",
            [
                build_labels(
                    factor_values=factor_values,
                    forward_returns=forward_returns,
                    unbuyable_at_entry=limit_up,
                    exclude_unbuyable=True,
                )
            ],
        )

        # 不清洗时因子"很有效"，清洗后真相是负相关
        assert dirty.ic_mean > 0.5
        assert clean.ic_mean < 0
        assert dirty.ic_mean - clean.ic_mean > 1.0


class TestLayerBacktest:
    def test_monotonic_factor(self) -> None:
        samples = build_labels(
            factor_values={A: 1.0, B: 2.0, C: 3.0, DE: 4.0},
            forward_returns={A: 0.01, B: 0.02, C: 0.03, DE: 0.04},
        )
        stats = layer_backtest("f", samples, layers=2)
        assert stats.is_monotonic
        assert stats.long_short_return > 0

    def test_non_monotonic_factor_detected(self) -> None:
        """中间乱序的因子多半是噪声。"""
        samples = build_labels(
            factor_values={A: 1.0, B: 2.0, C: 3.0, DE: 4.0},
            forward_returns={A: 0.01, B: 0.10, C: -0.05, DE: 0.04},
        )
        assert not layer_backtest("f", samples, layers=4).is_monotonic

    def test_too_few_samples__raises(self) -> None:
        samples = build_labels(factor_values={A: 1.0}, forward_returns={A: 0.1})
        with pytest.raises(ValueError, match="少于分组数"):
            layer_backtest("f", samples, layers=5)

    def test_invalid_layers__raises(self) -> None:
        samples = build_labels(factor_values={A: 1.0}, forward_returns={A: 0.1})
        with pytest.raises(ValueError, match="分组数必须"):
            layer_backtest("f", samples, layers=1)


class TestFactorPanel:
    def test_coverage(self) -> None:
        panel = FactorPanel(trade_date=dt.date(2026, 7, 24), name="f", values={A: 1.0})
        assert panel.coverage([A, B]) == 0.5
        assert panel.coverage([]) == 0.0
        assert len(panel) == 1
        assert panel.symbols == (A,)
