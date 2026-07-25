"""研究纪律与稳健性测试（M6）。

核心命题只有一句：**同一个 Sharpe，试 1 次和试 200 次的可信度完全不同。**
这套机制的价值在于把这个差别变成一个能拦住上线的数字。
"""

from __future__ import annotations

import datetime as dt
import json
import random
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from quantstock.backtest.robustness import (
    annualised_sharpe,
    cost_sensitivity,
    estimate_capacity,
    parameter_sensitivity,
)
from quantstock.backtest.trials import (
    DSR_FLOOR,
    PBO_CEILING,
    Trial,
    TrialLog,
    TrialRecorder,
    admission_check,
    deflated_sharpe_ratio,
    dsr_z_score,
    expected_max_sharpe,
    parameter_plateau,
    probability_of_backtest_overfitting,
)
from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import StrategyError
from quantstock.infra.money import money
from quantstock.infra.types import Side, Symbol
from quantstock.portfolio.covariance import (
    correlation_from_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)
from quantstock.reporting.review import (
    InterventionOutcome,
    analyse_intervention_value,
    build_deviation_report,
)

NOW = dt.datetime(2026, 7, 25, 18, 0, tzinfo=CST)
MAOTAI = Symbol("600519.SH")
CATL = Symbol("300750.SZ")


@pytest.fixture(autouse=True)
def _frozen() -> None:
    """固定时钟。"""
    set_clock(FrozenClock(NOW))


def trial(sharpe: float, *, n: int = 1, segment: str = "train", periods: int = 250) -> Trial:
    """构造一条试验记录。"""
    return Trial(
        trial_id=f"t{n}",
        strategy="momentum",
        params={"lookback": 20 * n},
        sharpe=sharpe,
        n_periods=periods,
        segment=segment,
    )


class TestTrialLog:
    """试验流水。"""

    def test_append_and_read(self, tmp_path: Path) -> None:
        log = TrialLog(tmp_path / "trials.jsonl")
        log.append(trial(1.2, n=1))
        log.append(trial(0.8, n=2))
        assert [t.sharpe for t in log] == [1.2, 0.8]

    def test_append_only_never_rewrites(self, tmp_path: Path) -> None:
        # 删掉失败的尝试会让 DSR 系统性偏乐观
        path = tmp_path / "trials.jsonl"
        log = TrialLog(path)
        log.append(trial(1.2, n=1))
        first = path.read_text(encoding="utf-8")
        log.append(trial(0.3, n=2))
        assert path.read_text(encoding="utf-8").startswith(first)

    def test_corrupt_line_skipped(self, tmp_path: Path) -> None:
        # 一行坏数据不该让整份研究记录不可读
        path = tmp_path / "trials.jsonl"
        log = TrialLog(path)
        log.append(trial(1.2, n=1))
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{ broken\n")
        log.append(trial(0.9, n=2))
        assert len(list(log)) == 2

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert list(TrialLog(tmp_path / "absent.jsonl")) == []

    def test_filter_by_strategy_and_segment(self, tmp_path: Path) -> None:
        log = TrialLog(tmp_path / "t.jsonl")
        log.append(trial(1.0, n=1, segment="train"))
        log.append(trial(0.9, n=2, segment="test"))
        assert log.count("momentum") == 2
        assert log.count("momentum", segment="test") == 1
        assert log.count("other") == 0

    def test_test_segment_use_is_detectable(self, tmp_path: Path) -> None:
        # 测试集一次性使用：反复在同一测试集上调参，它就变成了第二个训练集
        log = TrialLog(tmp_path / "t.jsonl")
        assert not log.test_segment_used("momentum")
        log.append(trial(0.9, n=1, segment="test"))
        assert log.test_segment_used("momentum")

    def test_recorder_makes_logging_the_default(self, tmp_path: Path) -> None:
        # 忘记记录的代价是 DSR 偏乐观，而那种错误没有任何症状
        log = TrialLog(tmp_path / "t.jsonl")
        recorder = TrialRecorder(log=log, strategy="momentum")
        for lookback in (10, 20, 30):
            recorder.record({"lookback": lookback}, sharpe=lookback / 20, n_periods=250)
        assert log.count("momentum") == 3

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        log = TrialLog(tmp_path / "t.jsonl")
        original = trial(1.5, n=1)
        log.append(original)
        restored = next(iter(log))
        assert restored.params == original.params
        assert restored.sharpe == original.sharpe

    def test_created_at_uses_injected_clock(self, tmp_path: Path) -> None:
        log = TrialLog(tmp_path / "t.jsonl")
        log.append(trial(1.0, n=1))
        payload = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").strip())
        assert payload["created_at"].startswith("2026-07-25")


class TestDeflatedSharpe:
    """折减 Sharpe。"""

    def test_single_trial_is_not_deflated(self) -> None:
        assert expected_max_sharpe(1) == 0.0

    def test_more_trials_raise_the_bar(self) -> None:
        # 试得越多，"白捡"的最大值越高
        assert expected_max_sharpe(1000) > expected_max_sharpe(100) > expected_max_sharpe(10)

    def test_same_sharpe_credible_once_worthless_after_200_tries(self) -> None:
        # 这就是整套机制要说明的那件事
        once = deflated_sharpe_ratio(1.5, n_trials=1, n_periods=250)
        many = deflated_sharpe_ratio(1.5, n_trials=200, n_periods=250)
        assert once > DSR_FLOOR
        assert many <= DSR_FLOOR

    def test_longer_sample_sharpens_the_conclusion(self) -> None:
        # 观测越多，结论越确定——**朝真相的方向**，不是无脑变好。
        # 基准是 expected_max_sharpe(20)≈1.90：高于它更确定是真的，低于它更确定是假的。
        assert deflated_sharpe_ratio(2.5, n_trials=20, n_periods=2000) > deflated_sharpe_ratio(
            2.5, n_trials=20, n_periods=60
        )
        # 反方向要用 z 值看——概率那一侧已经贴到 0 了（见下一个测试）
        assert dsr_z_score(0.5, n_trials=20, n_periods=2000) < dsr_z_score(
            0.5, n_trials=20, n_periods=60
        )

    def test_fat_tails_reduce_confidence(self) -> None:
        # 厚尾让 Sharpe 的不确定性变大。要在观测值高于基准时看——
        # 低于基准时"不确定性变大"反而会把结论往中性拉，方向相反
        normal = deflated_sharpe_ratio(2.5, n_trials=5, n_periods=250, kurtosis=3.0)
        fat = deflated_sharpe_ratio(2.5, n_trials=5, n_periods=250, kurtosis=15.0)
        assert fat < normal

    def test_probability_saturates_but_z_score_does_not(self) -> None:
        # 这就是保留 z 值的理由：60 期与 2000 期的 DSR 打印出来一模一样，
        # 而 z 值分别是 -10 与 -59，差着五倍多的确定性
        assert deflated_sharpe_ratio(0.5, n_trials=20, n_periods=60) == pytest.approx(
            deflated_sharpe_ratio(0.5, n_trials=20, n_periods=2000), abs=1e-12
        )
        assert (
            dsr_z_score(0.5, n_trials=20, n_periods=2000)
            < dsr_z_score(0.5, n_trials=20, n_periods=60)
            < 0
        )

    def test_too_few_periods_rejected(self) -> None:
        with pytest.raises(ValueError, match="至少需要 2 个样本期"):
            deflated_sharpe_ratio(1.0, n_trials=5, n_periods=1)

    def test_result_stays_in_range(self) -> None:
        assert -1.0 <= deflated_sharpe_ratio(5.0, n_trials=2, n_periods=250) <= 1.0
        assert -1.0 <= deflated_sharpe_ratio(-5.0, n_trials=500, n_periods=250) <= 1.0


class TestPBO:
    """过拟合概率（CSCV）。"""

    def test_pure_noise_is_near_a_coin_flip(self) -> None:
        # 全是噪声时，"选样本内最优"应该和随机选差不多
        rng = random.Random(7)
        matrix = [[rng.gauss(0, 0.01) for _ in range(120)] for _ in range(20)]
        pbo = probability_of_backtest_overfitting(matrix)
        assert 0.3 <= pbo <= 0.7

    def test_real_signal_gives_low_pbo(self) -> None:
        # 真有信号时，样本内最优在样本外也该最优
        rng = random.Random(7)
        matrix = [[rng.gauss(0.004, 0.01) for _ in range(120)]]
        matrix += [[rng.gauss(0, 0.01) for _ in range(120)] for _ in range(19)]
        assert probability_of_backtest_overfitting(matrix) < PBO_CEILING

    def test_odd_splits_rejected(self) -> None:
        matrix = [[0.01] * 40 for _ in range(5)]
        with pytest.raises(ValueError, match="偶数"):
            probability_of_backtest_overfitting(matrix, n_splits=5)

    def test_too_few_trials_rejected(self) -> None:
        with pytest.raises(ValueError, match="至少需要"):
            probability_of_backtest_overfitting([[0.01] * 40, [0.02] * 40])

    def test_ragged_matrix_rejected(self) -> None:
        with pytest.raises(ValueError, match="长度必须一致"):
            probability_of_backtest_overfitting([[0.01] * 40, [0.01] * 30, [0.01] * 40, [0.0] * 40])

    def test_too_few_periods_rejected(self) -> None:
        with pytest.raises(ValueError, match="不足以切成"):
            probability_of_backtest_overfitting([[0.01] * 8 for _ in range(5)], n_splits=8)


class TestParameterPlateau:
    """参数高原。"""

    def test_plateau_passes(self) -> None:
        assert parameter_plateau([0.8, 0.9, 1.0, 0.95, 0.85])

    def test_spike_fails(self) -> None:
        # 最优点很高但邻域全是负的——换个市场环境就完全失效
        assert not parameter_plateau([-0.2, -0.1, -0.3, -0.15])

    def test_empty_neighbourhood_fails(self) -> None:
        # 没检验过不等于通过
        assert not parameter_plateau([])

    def test_sensitivity_report_quantifies_the_shape(self) -> None:
        plateau = parameter_sensitivity(1.0, [0.9, 0.95, 0.85, 0.92])
        spike = parameter_sensitivity(1.0, [0.05, -0.1, 0.02, -0.05])

        assert plateau.is_plateau
        assert not spike.is_plateau
        assert spike.degradation > plateau.degradation
        assert "尖峰" in spike.explain()

    def test_empty_neighbourhood_raises(self) -> None:
        with pytest.raises(ValueError, match="不能为空"):
            parameter_sensitivity(1.0, [])


class TestAdmissionCheck:
    """实盘候选池准入。"""

    def _matrix(self, seed: int, *, signal: bool) -> list[list[float]]:
        rng = random.Random(seed)
        if signal:
            rows = [[rng.gauss(0.004, 0.01) for _ in range(120)]]
            rows += [[rng.gauss(0, 0.01) for _ in range(120)] for _ in range(9)]
            return rows
        return [[rng.gauss(0, 0.01) for _ in range(120)] for _ in range(10)]

    def test_no_trials_is_a_hard_error(self) -> None:
        # 没有记录就无法判断是否过拟合——这种情况必须拒绝而不是放行
        with pytest.raises(StrategyError, match="禁止入池"):
            admission_check([])

    def test_single_credible_trial_admitted(self) -> None:
        verdict = admission_check(
            [trial(1.8, n=1)],
            returns_matrix=self._matrix(1, signal=True),
            plateau=[1.5, 1.6, 1.7],
        )
        assert verdict.admitted
        assert verdict.dsr > DSR_FLOOR
        assert "允许" in verdict.explain()

    def test_two_hundred_trials_denied(self) -> None:
        # 同样的 Sharpe，试了 200 次就不可信了
        trials = [trial(0.3 + i * 0.006, n=i) for i in range(200)]
        verdict = admission_check(
            trials, returns_matrix=self._matrix(2, signal=False), plateau=[1.0, 1.1]
        )
        assert not verdict.admitted
        assert verdict.reasons, "拒绝必须说明理由"
        assert verdict.pbo > PBO_CEILING

    def test_missing_pbo_data_blocks_admission(self) -> None:
        # 没算 PBO 不等于 PBO 合格
        verdict = admission_check([trial(1.8, n=1)], plateau=[1.5, 1.6])
        assert not verdict.admitted
        assert any("PBO 无法计算" in r for r in verdict.reasons)

    def test_missing_plateau_blocks_admission(self) -> None:
        verdict = admission_check([trial(1.8, n=1)], returns_matrix=self._matrix(3, signal=True))
        assert not verdict.admitted
        assert not verdict.plateau_ok

    def test_spike_blocks_admission(self) -> None:
        verdict = admission_check(
            [trial(1.8, n=1)],
            returns_matrix=self._matrix(4, signal=True),
            plateau=[-0.5, -0.3, -0.4],
        )
        assert not verdict.admitted
        assert any("参数尖峰" in r for r in verdict.reasons)


class TestCostSensitivity:
    """成本敏感性。"""

    def test_robust_strategy_survives_double_cost(self) -> None:
        result = cost_sensitivity(gross_return=0.30, cost_per_turn=0.002, turnover=10)
        assert result.survives_double_cost
        assert result.breakeven_multiplier is None

    def test_fragile_strategy_dies_at_double_cost(self) -> None:
        # 只在零成本假设下成立的策略是不存在的策略
        result = cost_sensitivity(gross_return=0.05, cost_per_turn=0.002, turnover=15)
        assert not result.survives_double_cost
        assert result.breakeven_multiplier is not None
        assert "归零" in result.explain()

    def test_zero_cost_has_no_breakeven(self) -> None:
        result = cost_sensitivity(gross_return=0.1, cost_per_turn=0.0, turnover=10)
        assert result.breakeven_multiplier is None


class TestCapacity:
    """策略容量（A7）。"""

    def test_capacity_bound_by_least_liquid_name(self) -> None:
        # 组合里有一只买不进去，整个组合就建不起来
        estimate = estimate_capacity(
            weights={"600519.SH": 0.5, "300750.SZ": 0.5},
            adv={"600519.SH": money("5000000000"), "300750.SZ": money("10000000")},
            current_capital=money("1000000"),
        )
        assert estimate.binding_symbol == "300750.SZ"

    def test_utilisation_and_warning(self) -> None:
        estimate = estimate_capacity(
            weights={"600519.SH": 1.0},
            adv={"600519.SH": money("10000000")},
            current_capital=money("900000"),
        )
        assert estimate.utilisation > 0.8
        assert estimate.is_constrained
        assert "接近上限" in estimate.explain()

    def test_missing_adv_means_zero_capacity(self) -> None:
        # 宁可低估也不要高估——高估容量的代价是真金白银的冲击成本
        estimate = estimate_capacity(
            weights={"600519.SH": 1.0},
            adv={},
            current_capital=money("100000"),
        )
        assert estimate.max_capital == Decimal(0)
        assert estimate.utilisation == 1.0

    def test_higher_tolerance_gives_more_capacity(self) -> None:
        loose = estimate_capacity(
            weights={"600519.SH": 1.0},
            adv={"600519.SH": money("100000000")},
            current_capital=money("1000000"),
            impact_tolerance=0.004,
        )
        tight = estimate_capacity(
            weights={"600519.SH": 1.0},
            adv={"600519.SH": money("100000000")},
            current_capital=money("1000000"),
            impact_tolerance=0.001,
        )
        assert loose.max_capital > tight.max_capital

    def test_empty_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="权重为空"):
            estimate_capacity(weights={}, adv={}, current_capital=money("1"))

    def test_participation_limit_binds_for_liquid_names(self) -> None:
        # 冲击约束很松时，参与率上限成为约束
        estimate = estimate_capacity(
            weights={"600519.SH": 1.0},
            adv={"600519.SH": money("1000000")},
            current_capital=money("1"),
            impact_tolerance=0.5,
            participation_limit=0.05,
        )
        assert estimate.max_capital == money("50000")


class TestCovariance:
    """Ledoit-Wolf 收缩。"""

    def _returns(self, n_assets: int, n_obs: int, seed: int = 3) -> list[list[float]]:
        """等相关结构的收益：所有标的对同一个共同因子等载荷。"""
        rng = random.Random(seed)
        common = [rng.gauss(0, 0.01) for _ in range(n_obs)]
        return [
            [0.6 * common[t] + rng.gauss(0, 0.01) for t in range(n_obs)] for _ in range(n_assets)
        ]

    def _block_returns(self, n_assets: int, n_obs: int, seed: int = 5) -> list[list[float]]:
        """**非**等相关结构：两个板块，板块内高相关、板块间低相关。

        测"样本越多收缩越少"必须用这种数据。若真实结构本来就是等相关，
        样本越多只会越确信收缩目标是对的，δ 反而趋近 1——那是正确行为，
        不是 bug。
        """
        rng = random.Random(seed)
        market = [rng.gauss(0, 0.006) for _ in range(n_obs)]
        sector_a = [rng.gauss(0, 0.012) for _ in range(n_obs)]
        sector_b = [rng.gauss(0, 0.012) for _ in range(n_obs)]
        rows: list[list[float]] = []
        for k in range(n_assets):
            sector = sector_a if k < n_assets // 2 else sector_b
            rows.append(
                [0.3 * market[t] + 0.9 * sector[t] + rng.gauss(0, 0.004) for t in range(n_obs)]
            )
        return rows

    def test_sample_covariance_is_symmetric(self) -> None:
        cov = sample_covariance(self._returns(4, 100))
        for i in range(4):
            for j in range(4):
                assert cov[i][j] == pytest.approx(cov[j][i])

    def test_shrinkage_between_zero_and_one(self) -> None:
        estimate = ledoit_wolf_shrinkage(self._returns(5, 60))
        assert 0.0 <= estimate.shrinkage <= 1.0

    def test_small_sample_shrinks_harder(self) -> None:
        # 观测越少，样本协方差越不可信，越该靠向结构化先验。
        # 用板块结构数据：真实相关矩阵不是等相关，样本足够时应少收缩
        scarce = ledoit_wolf_shrinkage(self._block_returns(8, 20))
        plenty = ledoit_wolf_shrinkage(self._block_returns(8, 2000))
        assert scarce.shrinkage > plenty.shrinkage

    def test_correct_target_earns_full_shrinkage(self) -> None:
        # 真实结构就是等相关时，样本越多越该确信目标是对的
        estimate = ledoit_wolf_shrinkage(self._returns(8, 2000))
        assert estimate.shrinkage > 0.9

    def test_variances_are_preserved_on_the_diagonal(self) -> None:
        # 收缩目标的对角线就是样本方差，所以对角线不受收缩影响
        returns = self._returns(4, 100)
        sample = sample_covariance(returns)
        estimate = ledoit_wolf_shrinkage(returns)
        for i in range(4):
            assert estimate.variance_of(i) == pytest.approx(sample[i][i])

    def test_forced_shrinkage_of_one_gives_the_target(self) -> None:
        returns = self._returns(3, 80)
        estimate = ledoit_wolf_shrinkage(returns, forced_shrinkage=1.0)
        assert estimate.shrinkage == 1.0
        # 等相关目标下所有非对角元素的隐含相关系数相同
        corr = correlation_from_covariance(estimate.matrix)
        off_diagonal = [corr[0][1], corr[0][2], corr[1][2]]
        assert off_diagonal[0] == pytest.approx(off_diagonal[1])
        assert off_diagonal[1] == pytest.approx(off_diagonal[2])

    def test_forced_shrinkage_of_zero_gives_the_sample(self) -> None:
        returns = self._returns(3, 80)
        sample = sample_covariance(returns)
        estimate = ledoit_wolf_shrinkage(returns, forced_shrinkage=0.0)
        assert estimate.matrix[0][1] == pytest.approx(sample[0][1])

    def test_out_of_range_shrinkage_rejected(self) -> None:
        with pytest.raises(ValueError, match="必须在 0~1"):
            ledoit_wolf_shrinkage(self._returns(3, 50), forced_shrinkage=1.5)

    def test_ragged_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="长度必须一致"):
            sample_covariance([[0.1, 0.2], [0.1]])

    def test_too_few_observations_rejected(self) -> None:
        with pytest.raises(ValueError, match="至少需要"):
            sample_covariance([[0.1], [0.2]])

    def test_correlation_diagonal_is_one(self) -> None:
        corr = correlation_from_covariance(sample_covariance(self._returns(3, 60)))
        for i in range(3):
            assert corr[i][i] == pytest.approx(1.0)

    def test_zero_variance_asset_yields_zero_correlation(self) -> None:
        # 不该产生 NaN 或除零
        corr = correlation_from_covariance([[0.0, 0.0], [0.0, 0.04]])
        assert corr[0][0] == 0.0
        assert corr[0][1] == 0.0

    def test_heavy_shrinkage_flag(self) -> None:
        estimate = ledoit_wolf_shrinkage(self._returns(3, 50), forced_shrinkage=0.8)
        assert estimate.is_heavily_shrunk
        assert "不可尽信" in estimate.explain()


class TestInterventionValue:
    """人工干预价值分析（D3）。"""

    def _outcome(self, reason: str, *, later: str, side: Side = Side.BUY) -> InterventionOutcome:
        return InterventionOutcome(
            intent_id="i1",
            symbol=MAOTAI,
            side=side,
            skip_reason=reason,
            suggested_price=money("100"),
            later_price=money(later),
            horizon_days=20,
        )

    def test_skipping_a_buy_that_rose_was_wrong(self) -> None:
        outcome = self._outcome("disagree_logic", later="110")
        assert outcome.forgone_return > 0
        assert not outcome.skip_was_right

    def test_skipping_a_buy_that_fell_was_right(self) -> None:
        outcome = self._outcome("disagree_logic", later="90")
        assert outcome.skip_was_right

    def test_sell_side_sign_is_flipped(self) -> None:
        # 跳过卖出建议后股价下跌 → 错过了避损 → 跳错了
        outcome = self._outcome("bad_timing", later="90", side=Side.SELL)
        assert outcome.forgone_return > 0
        assert not outcome.skip_was_right

    def test_grouped_by_reason(self) -> None:
        outcomes = [
            self._outcome("disagree_logic", later="110"),
            self._outcome("disagree_logic", later="105"),
            self._outcome("cash_reserved", later="95"),
        ]
        groups = analyse_intervention_value(outcomes)
        assert [g.reason for g in groups] == ["disagree_logic", "cash_reserved"]
        assert groups[0].count == 2

    def test_small_sample_refuses_to_conclude(self) -> None:
        # 用 3 次跳过论证"该不该相信程序"得到的是噪声
        groups = analyse_intervention_value([self._outcome("bad_timing", later="110")] * 3)
        assert not groups[0].has_enough_samples
        assert "不足以下结论" in groups[0].verdict

    def test_consistently_losing_intervention_flagged(self) -> None:
        groups = analyse_intervention_value([self._outcome("disagree_logic", later="110")] * 12)
        assert groups[0].has_enough_samples
        assert "跑输" in groups[0].verdict
        assert "减少此类跳过" in groups[0].explain()

    def test_consistently_winning_intervention_flagged(self) -> None:
        # 长期跑赢程序意味着策略有系统性缺陷，该改的是策略
        groups = analyse_intervention_value([self._outcome("disagree_logic", later="90")] * 12)
        assert "跑赢" in groups[0].verdict
        assert "系统性缺陷" in groups[0].verdict

    def test_zero_suggested_price_is_safe(self) -> None:
        outcome = InterventionOutcome(
            intent_id="i1",
            symbol=CATL,
            side=Side.BUY,
            skip_reason="other",
            suggested_price=money("0"),
            later_price=money("10"),
            horizon_days=5,
        )
        assert outcome.forgone_return == Decimal(0)


class TestDeviationReport:
    """计划-实际偏差。"""

    def test_full_execution(self) -> None:
        report = build_deviation_report(
            trade_date="2026-07-24",
            planned=3,
            executed=3,
            skipped=0,
            planned_amount=money("100000"),
            executed_amount=money("99000"),
        )
        assert report.execution_rate == 1.0
        assert not report.needs_attention
        assert report.amount_drift < 0

    def test_low_execution_rate_needs_attention(self) -> None:
        # 执行率长期偏低说明建议与用户判断系统性不一致
        report = build_deviation_report(
            trade_date="2026-07-24",
            planned=10,
            executed=2,
            skipped=8,
            planned_amount=money("100000"),
            executed_amount=money("20000"),
            by_reason={"disagree_logic": 8},
        )
        assert report.needs_attention
        assert "disagree_logic×8" in report.explain()

    def test_aborted_plan(self) -> None:
        report = build_deviation_report(
            trade_date="2026-07-24",
            planned=5,
            executed=0,
            skipped=0,
            planned_amount=money("100000"),
            executed_amount=money("0"),
            aborted=True,
        )
        assert report.needs_attention
        assert "中止" in report.explain()

    def test_empty_plan_is_safe(self) -> None:
        report = build_deviation_report(
            trade_date="2026-07-24",
            planned=0,
            executed=0,
            skipped=0,
            planned_amount=money("0"),
            executed_amount=money("0"),
        )
        assert report.execution_rate == 0.0
        assert report.amount_drift == Decimal(0)
        assert not report.needs_attention


def test_annualised_sharpe() -> None:
    """年化 Sharpe。"""
    assert annualised_sharpe([0.001] * 252) == 0.0, "零波动应返回 0 而不是无穷大"
    assert annualised_sharpe([0.01]) == 0.0
    rng = random.Random(11)
    positive = [rng.gauss(0.001, 0.01) for _ in range(252)]
    assert annualised_sharpe(positive) > 0


class TestNoCircularImports:
    """包级循环导入回归测试。

    ``quantstock.portfolio`` 曾经无法作为首个导入使用：
    portfolio → account → risk.costs → risk/__init__ → risk.engine → portfolio，
    一个完整的包级环。测试套件当时没发现，因为别的测试先导入了 `risk`，
    包已经在 ``sys.modules`` 里了。

    只有**在干净的解释器里逐个导入**才能暴露这类问题。
    """

    @pytest.mark.parametrize(
        "module",
        [
            "quantstock.account",
            "quantstock.advisor",
            "quantstock.backtest",
            "quantstock.cli",
            "quantstock.costs",
            "quantstock.data",
            "quantstock.execution",
            "quantstock.factors",
            "quantstock.intel",
            "quantstock.llm",
            "quantstock.portfolio",
            "quantstock.reporting",
            "quantstock.risk",
            "quantstock.services",
            "quantstock.strategy",
            "quantstock.web",
        ],
    )
    def test_package_imports_in_a_fresh_interpreter(self, module: str) -> None:
        result = subprocess.run(  # noqa: S603 - 参数来自本测试的固定列表
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{module} 无法独立导入：\n{result.stderr}"
