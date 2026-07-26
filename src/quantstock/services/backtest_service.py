"""回测服务：把同一套决策逻辑放到历史区间上跑。

**关键要求（用户明确提出的）**：出每日建议的那条逻辑，必须能选定历史区间回测。
所以这里**复用 ``AdvisorService`` 的打分与组合逻辑**，而不是另写一套回测策略——
两套逻辑迟早会分叉，那时回测结果就不再说明实盘会怎样了。

三条硬约束：

- **LLM 强制 replay**（红线 LR3）：由 ``LLMService(in_backtest=True)`` 保证，
  任何实时调用都会抛 ``LLMLiveCallInBacktestError``；
- **PIT**：引擎的 ``MarketView`` 只暴露 ``as_of`` 及之前的 bar（红线 R2）；
- **试验记录**：每次回测都写进 ``trials.jsonl``，删掉失败尝试会让 DSR 偏乐观。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from quantstock.account.ledger import Ledger
from quantstock.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    MarketView,
    Order,
    StrategyFn,
)
from quantstock.backtest.metrics import PerformanceStats
from quantstock.backtest.trials import AdmissionVerdict, Trial, TrialLog, admission_check
from quantstock.config.settings import Settings
from quantstock.infra.clock import now
from quantstock.infra.errors import DataError
from quantstock.infra.logging import get_logger
from quantstock.infra.money import money
from quantstock.infra.types import Money, Symbol, TradeDate
from quantstock.portfolio.builder import build_targets, diff_to_orders
from quantstock.services.advisor_service import (
    DEFAULT_STRATEGY_WEIGHTS,
    MIN_HISTORY_BARS,
    AdvisorService,
    constraints_from,
)
from quantstock.services.data_service import DataService
from quantstock.strategy.builtin import (
    EtfRotationStrategy,
    MacroExposureStrategy,
    MomentumTrendStrategy,
    TimingOverlayStrategy,
    blend_scores,
)
from quantstock.strategy.types import Strategy, StrategyContext

# 界面与 CLI 是"薄"客户端，只允许依赖 services（F20.1 分层契约）。
# 过拟合防线的契约类型在这里转出，客户端不必、也不允许直接 import backtest 层。
__all__ = ["AdmissionVerdict", "BacktestReport", "BacktestService", "Trial"]

_log = get_logger(__name__)

DEFAULT_BACKTEST_CASH = money("200000")
"""回测默认初始资金。取一个个人账户的典型量级，而不是引擎默认的一百万——
资金量会影响能不能买满一手，用一百万回测再拿十万实盘，结论对不上。"""

DEFAULT_REBALANCE_DAYS = 5
"""默认调仓间隔（交易日）。

每天调仓在扣掉成本后几乎必然是负贡献——A 股单边成本约 0.1%，
日频换手一年就是几十个点。默认一周一次，具体最优频率由
因子衰减曲线决定（docs/08 A6）。
"""


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """一次回测的结果。"""

    start: TradeDate
    end: TradeDate
    trading_days: int
    stats: PerformanceStats
    final_equity: float
    initial_cash: float
    fills: int
    rejections: dict[str, int]
    trial_id: str
    universe: tuple[Symbol, ...]
    llm_mode: str

    @property
    def total_return(self) -> float:
        """区间总收益。"""
        return self.stats.total_return

    def explain(self) -> str:
        """人类可读结论。

        Returns:
            结论文本。
        """
        return (
            f"{self.start} ~ {self.end}（{self.trading_days} 个交易日）："
            f"总收益 {self.stats.total_return:+.2%}，年化 {self.stats.annualized_return:+.2%}，"
            f"Sharpe {self.stats.sharpe:.2f}，最大回撤 {self.stats.max_drawdown:.2%}，"
            f"成交 {self.fills} 笔"
        )

    def warnings(self) -> list[str]:
        """需要提醒用户的地方。

        Returns:
            提醒列表。**回测跑出来好看不等于策略好**，这些提示就是为了
            让人别把一次回测当结论。
        """
        notes: list[str] = []
        if self.trading_days < 250:  # noqa: PLR2004 - 不足一年
            notes.append(f"区间仅 {self.trading_days} 个交易日，不足一年，结论不具代表性")
        if self.fills == 0:
            notes.append("零成交——请检查候选池、调仓频率与资金是否够买一手")
        if self.rejections:
            top = max(self.rejections.items(), key=lambda kv: kv[1])
            notes.append(f"拒单最多的原因是 {top[0]}（{top[1]} 次）")
        if self.llm_mode == "replay":
            notes.append("LLM 走缓存回放；缓存未命中的决策点等同纯量化")
        return notes


class BacktestService:
    """回测编排。"""

    def __init__(
        self,
        settings: Settings,
        *,
        data: DataService | None = None,
    ) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
            data: 数据服务。
        """
        self._settings = settings
        self._data = data or DataService(settings)
        self._trials = TrialLog(settings.var_dir / "research" / "trials.jsonl")
        self._constraints = constraints_from(settings)

    @property
    def trials(self) -> TrialLog:
        """试验流水。"""
        return self._trials

    def run(
        self,
        *,
        start: TradeDate,
        end: TradeDate,
        universe: Sequence[Symbol] | None = None,
        tier: str = "core",
        initial_cash: Money | None = None,
        rebalance_days: int = DEFAULT_REBALANCE_DAYS,
        segment: str = "train",
        record: bool = True,
    ) -> BacktestReport:
        """在历史区间上回测每日建议逻辑。

        Args:
            start: 起始日。
            end: 结束日。
            universe: 候选池；None 表示按档位解析。
            tier: 候选池档位。
            initial_cash: 初始资金。
            rebalance_days: 调仓间隔（交易日）。
            segment: 数据段 ``train`` / ``validation`` / ``test``。
                **test 段每个策略只允许跑一次**——反复在测试集上调参，
                它就变成了第二个训练集。
            record: 是否记入试验流水。

        Returns:
            回测报告。

        Raises:
            DataError: 区间内数据不足。
        """
        symbols = tuple(universe or self._data.resolve_universe(tier))
        # 多读一段前置历史：起始日当天就要能算 MA60，否则开头几十天全是空信号
        warmup = start - dt.timedelta(days=MIN_HISTORY_BARS * 2)
        history = self._data.read_bars(symbols, start=warmup, end=end)
        if not history:
            msg = "回测区间内没有行情数据，请先执行 quantstock data init"
            raise DataError(msg, start=start.isoformat(), end=end.isoformat())

        trading_days = sorted(
            {
                bar.trade_date
                for bars in history.values()
                for bar in bars
                if start <= bar.trade_date <= end
            }
        )
        if not trading_days:
            msg = "回测区间内没有交易日"
            raise DataError(msg, start=start.isoformat(), end=end.isoformat())

        cash = initial_cash or DEFAULT_BACKTEST_CASH
        engine = BacktestEngine(config=BacktestConfig(initial_cash=cash))

        result = engine.run(
            strategy=self._make_strategy(symbols, rebalance_days=rebalance_days),
            history=history,
            trading_days=trading_days,
        )

        trial_id = ""
        if record:
            trial_id = self._record(result.stats, start=start, end=end, segment=segment)

        report = BacktestReport(
            start=trading_days[0],
            end=trading_days[-1],
            trading_days=len(trading_days),
            stats=result.stats,
            final_equity=result.equity[-1] if result.equity else float(cash),
            initial_cash=float(cash),
            fills=len(result.fills),
            rejections=result.rejection_summary,
            trial_id=trial_id,
            universe=symbols,
            llm_mode="replay" if self._settings.config.llm.enabled else "off",
        )
        _log.info(
            "backtest_done",
            start=start.isoformat(),
            end=end.isoformat(),
            days=len(trading_days),
            sharpe=round(result.stats.sharpe, 3),
            fills=len(result.fills),
        )
        return report

    def _make_strategy(self, universe: Sequence[Symbol], *, rebalance_days: int) -> StrategyFn:
        """构造回测用的策略函数。

        **这里跑的就是 ``AdvisorService`` 的那套打分与组合逻辑**：
        多策略融合 → 择时系数 → 目标权重 → 差分调仓。
        唯一的差别是不生成四支柱解释（回测里没人读），也不过风控引擎的
        人工确认环节。

        Args:
            universe: 候选池。
            rebalance_days: 调仓间隔。

        Returns:
            引擎可用的策略函数。
        """
        strategies: dict[str, Strategy] = {
            "momentum_trend": MomentumTrendStrategy(),
            "etf_rotation": EtfRotationStrategy(),
        }
        timing_overlay = TimingOverlayStrategy()
        macro = MacroExposureStrategy()
        constraints = self._constraints
        counter = {"day": 0}

        def strategy(view: MarketView, ledger: Ledger) -> Sequence[Order]:
            counter["day"] += 1
            if counter["day"] % rebalance_days != 0:
                return []  # 非调仓日不动，避免日频换手把收益吃光

            tradable = tuple(s for s in universe if len(view.bars(s)) >= MIN_HISTORY_BARS)
            if not tradable:
                return []

            ctx = StrategyContext(as_of=view.as_of, market=view, universe=tradable)
            scores = blend_scores(
                {name: list(impl.generate(ctx)) for name, impl in strategies.items()},
                DEFAULT_STRATEGY_WEIGHTS,
            )
            prices = {s: bar.close for s in tradable if (bar := view.latest(s)) is not None}
            if not prices:
                return []

            positions = ledger.positions()
            total_value = ledger.cash + sum(
                (prices.get(s, Decimal(0)) * p.qty for s, p in positions.items()),
                start=Decimal(0),
            )

            targets = build_targets(
                scores=scores,
                prices=prices,
                total_value=total_value,
                exposure=macro.generate_exposure(ctx).target_exposure,
                constraints=constraints,
                timing={
                    s: sig.coefficient for s, sig in timing_overlay.generate_timing(ctx).items()
                },
            )
            orders, _ = diff_to_orders(
                targets=targets,
                positions=positions,
                prices=prices,
                total_value=total_value,
                constraints=constraints,
            )
            return [
                Order(symbol=o.symbol, side=o.side, qty=abs(o.qty), reason=o.reason)
                for o in orders
                if o.qty != 0
            ]

        return strategy

    def _record(
        self,
        stats: PerformanceStats,
        *,
        start: TradeDate,
        end: TradeDate,
        segment: str,
    ) -> str:
        """把本次回测记入试验流水。

        **每次尝试都要记**，包括失败的。只留最优结果会让 DSR 系统性偏乐观，
        而 DSR 正是用来判断"这个结果是真的好还是试出来的"。

        Args:
            stats: 绩效指标。
            start: 起始日。
            end: 结束日。
            segment: 数据段。

        Returns:
            试验 ID。
        """
        existing = self._trials.count("daily_advice", segment=segment)
        trial = Trial(
            trial_id=f"daily_advice-{segment}-{existing + 1:04d}",
            strategy="daily_advice",
            params={
                "weights": DEFAULT_STRATEGY_WEIGHTS,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "max_holdings": self._constraints.max_holdings,
            },
            sharpe=stats.sharpe,
            annual_return=stats.annualized_return,
            max_drawdown=stats.max_drawdown,
            n_periods=stats.trading_days,
            segment=segment,
            created_at=now().isoformat(),
        )
        self._trials.append(trial)
        return trial.trial_id

    def trial_records(
        self, strategy: str = "daily_advice", *, segment: str | None = None
    ) -> list[Trial]:
        """取某策略的试验记录。

        Args:
            strategy: 策略名。
            segment: 数据段；None 表示全部。

        Returns:
            试验记录，按写入顺序。

        """
        return self._trials.for_strategy(strategy, segment=segment)

    def admission(self, strategy: str = "daily_advice") -> AdmissionVerdict:
        """实盘候选池准入检查（A5 强制门槛）。

        **必须喂全部试验记录**，包括失败的。只留最优结果会让 DSR 系统性偏乐观，
        而 DSR 正是用来判断"这个结果是真的好还是试出来的"。

        本方法刻意不接受"只看某几次试验"的参数——那等于把删掉失败尝试
        做成了一个功能。

        Args:
            strategy: 策略名。

        Returns:
            准入结论。DSR < 0.95 或 PBO > 0.5 时 ``admitted`` 为 False，
            且 ``reasons`` 说明具体是哪一项没过。

        Raises:
            StrategyError: 没有任何试验记录。
        """
        return admission_check(self._trials.for_strategy(strategy))

    def advisor(self) -> AdvisorService:
        """构造一个**回测模式**的建议服务。

        用于"某一天当时会给出什么建议"这类回溯，LLM 被强制为 replay。

        Returns:
            建议服务。
        """
        from quantstock.services.llm_service import LLMService  # noqa: PLC0415 - 避免循环

        return AdvisorService(
            self._settings,
            data=self._data,
            llm=LLMService(self._settings, in_backtest=True),
        )
