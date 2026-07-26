"""建议服务：把每日决策链路真正串起来。

**这是之前缺的那根脊椎。** 在它出现之前，每一层都单独可用、单独测过，
但没有任何代码把它们连起来跑成"今日建议"——``PlanBuilder`` 全项目无人调用，
用户拿不到一条建议。1000 多个测试全绿也掩盖不了这件事，因为每层的测试
都只在测自己那一层。

链路（对应 docs/02 第四节的每日时序）::

    ① 数据      从数据湖读 PIT 安全的历史行情
    ② 因子/技术 build_analytics（支柱②）
    ③ 信号      多策略打分 → blend_scores → base_score   ← 纯量化
    ④ 研判      [L2 LLM] conviction_adjustment            ← 有界，可关
    ⑤ 组合      build_targets → diff_to_orders
    ⑥ 风控      RiskEngine.pre_trade_check + 情报黑名单    ← LLM 影响不到
    ⑦ 建议      PlanBuilder → 四支柱 → PlanStore 落盘

两条不可让步的约束贯穿全程：

- **PIT**：只用 ``as_of`` 当日及之前的 bar（红线 R2）；
- **LLM 有界**：④ 只能通过 ``conviction_adjustment`` 一个出口影响 ③ 的打分，
  且 ⑤⑥ 完全在它的影响范围之外（红线 LR1）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from quantstock.account.ledger import Ledger
from quantstock.advisor.analytics import build_analytics
from quantstock.advisor.planner import PlanBuilder, compute_param_hash
from quantstock.advisor.store import PlanStore
from quantstock.advisor.types import (
    IntelEvidence,
    IntelImpact,
    PositionAnalytics,
    RationaleBundle,
    TradePlan,
)
from quantstock.backtest.engine import MarketView
from quantstock.config.settings import Settings
from quantstock.data.types import Bar
from quantstock.infra.clock import CST, now, today
from quantstock.infra.errors import DataError
from quantstock.infra.logging import get_logger
from quantstock.infra.money import money
from quantstock.infra.types import Money, Symbol, TradeDate
from quantstock.portfolio.builder import PortfolioConstraints, build_targets, diff_to_orders
from quantstock.risk.engine import MarketSnapshot, RiskEngine
from quantstock.services.account_service import AccountService
from quantstock.services.data_service import DataService
from quantstock.services.intel_service import IntelService
from quantstock.services.llm_service import LLMService
from quantstock.strategy.builtin import (
    EtfRotationStrategy,
    MacroExposureStrategy,
    MomentumTrendStrategy,
    TimingOverlayStrategy,
    blend_scores,
)
from quantstock.strategy.types import Evidence, Signal, Strategy, StrategyContext

# CLI 与界面只允许依赖 services（F20.1 分层契约），计划契约在这里转出
__all__ = ["AdviceResult", "AdvisorService", "RationaleBundle", "TradePlan"]

_log = get_logger(__name__)

MIN_HISTORY_BARS = 60
"""出信号所需的最少 bar 数。

不足这个长度时 MA60 之类的指标算不出来，硬算会得到一个用短序列
凑出来的假均线——那比没有信号更危险。
"""

DEFAULT_STRATEGY_WEIGHTS: dict[str, float] = {"momentum_trend": 0.6, "etf_rotation": 0.4}
"""多策略融合权重。

暂为常量而非配置项：权重要改就得重跑 A/B 回测证明新权重更好，
把它做成界面上随手可调的滑块，等于鼓励在没有证据的情况下调参。
"""

DEFAULT_ACCOUNT_ID = "main"
"""默认账户标识。多账户支持尚未实现，先固定一个。"""

DEFAULT_COLD_START_CAPITAL = money("100000")
"""冷启动时假设的总资产。

没有账本的新用户也该能看到一份完整建议——看不到就无从判断这套系统
是否值得录入真实持仓。
"""

DEFAULT_LOOKBACK_DAYS = 400
"""读取历史的自然日窗口。约 260 个交易日，够算年线与 252 日分位。"""


@dataclass(frozen=True, slots=True)
class AdviceResult:
    """一次建议生成的完整结果。"""

    plan: TradePlan
    saved_to: str
    base_scores: dict[Symbol, float]
    final_scores: dict[Symbol, float]
    llm_used: bool
    llm_notes: dict[Symbol, str] = field(default_factory=dict)
    skipped: tuple[tuple[Symbol, str], ...] = ()
    """被组合层跳过的项，必须在日报中展示。"""
    data_gaps: tuple[Symbol, str] = ()  # type: ignore[assignment]

    @property
    def summary(self) -> str:
        """人类可读摘要。"""
        parts = [f"{len(self.plan.intents)} 条建议"]
        if self.plan.rejected:
            parts.append(f"风控否决 {len(self.plan.rejected)} 条")
        if self.plan.incomplete:
            parts.append(f"解释不完整剔除 {len(self.plan.incomplete)} 条")
        if self.skipped:
            parts.append(f"组合层跳过 {len(self.skipped)} 条")
        parts.append("🤖 LLM 参与" if self.llm_used else "纯量化")
        return "，".join(parts)


class AdvisorService:
    """每日建议编排。"""

    def __init__(
        self,
        settings: Settings,
        *,
        data: DataService | None = None,
        intel: IntelService | None = None,
        llm: LLMService | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
            data: 数据服务。
            intel: 情报服务。
            llm: 大模型服务。默认按配置构造（默认关闭）。
            ledger: 账本。None 表示**从流水重放**，没有流水时才是冷启动。

                这里曾经是个真实的缺陷：默认恒为 None，于是系统永远按空账户
                出建议——不知道你持有什么，因此支柱②没有真实成本与持有期，
                "再持有 N 天免红利税"永远不出现，卖出建议一条也生不成。
        """
        self._settings = settings
        self._data = data or DataService(settings)
        self._intel = intel or IntelService(settings)
        self._llm = llm or LLMService(settings)
        self._account = AccountService(settings)
        self._ledger = ledger if ledger is not None else self._account.ledger()
        self._store = PlanStore(settings.var_dir / "plans")
        self._constraints = _constraints_from(settings)
        self._risk = RiskEngine(constraints=self._constraints)

    @property
    def store(self) -> PlanStore:
        """计划仓库。"""
        return self._store

    def advise(
        self,
        *,
        as_of: TradeDate | None = None,
        universe: Sequence[Symbol] | None = None,
        tier: str | None = None,
        total_value: Money | None = None,
        cash: Money | None = None,
        exposure: Decimal | None = None,
        save: bool = True,
    ) -> AdviceResult:
        """生成今日建议。

        Args:
            as_of: 决策日；None 表示今日。
            universe: 候选池；None 表示按 ``tier`` 解析。
            tier: 候选池档位 ``core`` / ``all``；None 表示用配置里的 ``data.init_tier``。
            total_value: 账户总资产；None 时由账本推导，无账本则用配置的初始资金。
            cash: 可用资金；None 时同上。
            exposure: 总仓位中枢；None 表示由 LONG 层策略给出。
            save: 是否落盘。

        Returns:
            建议结果。

        Raises:
            DataError: 数据不足以出建议。
        """
        trade_date = as_of or today()
        symbols = tuple(
            universe or self._data.resolve_universe(tier or self._settings.config.data.init_tier)
        )

        history = self._load_history(symbols, as_of=trade_date)
        if not history:
            msg = "数据湖里没有可用行情，无法出建议。请先执行 quantstock data init"
            raise DataError(msg, as_of=trade_date.isoformat())

        view = MarketView(history, trade_date)
        tradable = tuple(s for s in symbols if len(history.get(s, [])) >= MIN_HISTORY_BARS)
        if not tradable:
            msg = f"所有标的的历史都不足 {MIN_HISTORY_BARS} 根 K 线，无法计算指标"
            raise DataError(msg, as_of=trade_date.isoformat(), symbols=len(symbols))

        # ③ 纯量化打分。LLM 在这一步完全不参与
        signals, base_scores = self._score(view, tradable, trade_date)
        prices = self._latest_prices(history, tradable)
        analytics = self._analytics(history, tradable, trade_date)

        # ④ L2 研判：唯一的数值出口，且被 α 限幅
        final_scores, llm_notes, llm_used = self._apply_llm(
            base_scores, analytics=analytics, as_of=trade_date
        )

        value = total_value or self._account_value(prices)
        available = cash if cash is not None else self._account_cash(value)
        centre = exposure if exposure is not None else self._exposure(view, tradable, trade_date)

        # ⑤ 组合
        constraints = self._constraints
        targets = build_targets(
            scores=final_scores,
            prices=prices,
            total_value=value,
            exposure=centre,
            constraints=constraints,
            timing=self._timing(view, tradable, trade_date),
        )
        positions = self._ledger.positions() if self._ledger else {}
        orders, skipped = diff_to_orders(
            targets=targets,
            positions=positions,
            prices=prices,
            total_value=value,
            constraints=constraints,
        )

        # ⑥ 风控。情报黑名单在这里汇入（通路 1），LLM 影响不到这一层
        decision = self._risk.pre_trade_check(
            orders=orders,
            positions=positions,
            cash=available,
            total_value=value,
            market=MarketSnapshot(
                bars={s: history[s][-1] for s in tradable if history.get(s)},
                blacklist=frozenset(e.symbol for e in self._intel.blacklist_entries(as_of=now())),
            ),
            trade_date=trade_date,
        )

        # ⑦ 建议
        plan = PlanBuilder(
            account_id=DEFAULT_ACCOUNT_ID,
            require_full_rationale=self._settings.config.advisor.require_full_rationale,
            price_band=Decimal(str(self._settings.config.advisor.price_band_pct)),
        ).build(
            trade_date=trade_date,
            decision=decision,
            analytics=analytics,
            quant_evidence=self._evidence(signals),
            counter_evidence=self._counter_evidence(signals, final_scores),
            intel=self._intel_evidence(tradable, as_of=trade_date),
            data_fingerprint=self._fingerprint(history),
            strategy_versions=self._strategy_versions(),
            param_hash=compute_param_hash(self._param_snapshot()),
        )

        saved = ""
        if save:
            saved = str(self._store.save(plan))

        _log.info(
            "advice_generated",
            trade_date=trade_date.isoformat(),
            universe=len(tradable),
            intents=len(plan.intents),
            llm=llm_used,
        )
        return AdviceResult(
            plan=plan,
            saved_to=saved,
            base_scores=base_scores,
            final_scores=final_scores,
            llm_used=llm_used,
            llm_notes=llm_notes,
            skipped=tuple((s.symbol, s.reason) for s in skipped),
        )

    # ------------------------------------------------------------------ 各段
    def _load_history(
        self, symbols: Sequence[Symbol], *, as_of: TradeDate
    ) -> dict[Symbol, list[Bar]]:
        """读取 PIT 安全的历史。

        **上界是 ``as_of``**——这是防未来函数的第一道闸（红线 R2）。

        Args:
            symbols: 标的列表。
            as_of: 决策日。

        Returns:
            标的 → K 线列表。
        """
        start = as_of - dt.timedelta(days=DEFAULT_LOOKBACK_DAYS)
        return self._data.read_bars(symbols, start=start, end=as_of)

    def _score(
        self, view: MarketView, universe: Sequence[Symbol], as_of: TradeDate
    ) -> tuple[dict[str, list[Signal]], dict[Symbol, float]]:
        """多策略打分与融合。

        Args:
            view: PIT 安全的行情视图。
            universe: 候选池。
            as_of: 决策日。

        Returns:
            ``(各策略信号, 融合后的 base_score)``。
        """
        ctx = StrategyContext(as_of=as_of, market=view, universe=tuple(universe))
        strategies: dict[str, Strategy] = {
            "momentum_trend": MomentumTrendStrategy(),
            "etf_rotation": EtfRotationStrategy(),
        }
        signals = {name: list(strategy.generate(ctx)) for name, strategy in strategies.items()}
        return signals, blend_scores(signals, DEFAULT_STRATEGY_WEIGHTS)

    def _timing(
        self, view: MarketView, universe: Sequence[Symbol], as_of: TradeDate
    ) -> dict[Symbol, Decimal]:
        """SHORT 层择时系数。

        Args:
            view: 行情视图。
            universe: 候选池。
            as_of: 决策日。

        Returns:
            标的 → 择时系数（0.5~1.0）。
        """
        ctx = StrategyContext(as_of=as_of, market=view, universe=tuple(universe))
        return {
            symbol: signal.coefficient
            for symbol, signal in TimingOverlayStrategy().generate_timing(ctx).items()
        }

    def _exposure(self, view: MarketView, universe: Sequence[Symbol], as_of: TradeDate) -> Decimal:
        """LONG 层总仓位中枢。

        Args:
            view: 行情视图。
            universe: 候选池。
            as_of: 决策日。

        Returns:
            0~1 的仓位中枢。
        """
        ctx = StrategyContext(as_of=as_of, market=view, universe=tuple(universe))
        return MacroExposureStrategy().generate_exposure(ctx).target_exposure

    def _apply_llm(
        self,
        base_scores: Mapping[Symbol, float],
        *,
        analytics: Mapping[Symbol, PositionAnalytics],
        as_of: TradeDate,
    ) -> tuple[dict[Symbol, float], dict[Symbol, str], bool]:
        """施加 L2 研判的有界影响。

        LLM 关闭、失效、超预算、缓存未命中——任一情况都原样返回 base_score，
        系统退化为纯量化并照常出建议（红线 LR2）。

        Args:
            base_scores: 纯量化打分。
            analytics: 技术分析，作为研判材料。
            as_of: 决策日。

        Returns:
            ``(final_scores, 各标的的 LLM 说明, 是否实际用上 LLM)``。
        """
        if not self._llm.task_enabled("position_judge"):
            return dict(base_scores), {}, False

        task = self._llm.position_judge()
        anonymizer = self._llm.anonymizer()
        final: dict[Symbol, float] = {}
        notes: dict[Symbol, str] = {}
        used = False

        for symbol, base in base_scores.items():
            materials = self._materials(symbol, analytics.get(symbol), as_of=as_of)
            outcome = task.run(
                symbol,
                base_score=base,
                materials=materials,
                as_of=as_of.isoformat(),
                anonymizer=anonymizer,
            )
            final[symbol] = outcome.influence.final_score
            notes[symbol] = outcome.influence.explain()
            used = used or outcome.used_llm
        return final, notes, used

    def _materials(
        self, symbol: Symbol, analytics: PositionAnalytics | None, *, as_of: TradeDate
    ) -> dict[str, str]:
        """组装研判材料。

        材料 ID 是 ``evidence_ref`` 的取值域——模型引用不存在的 ID
        会让整个输出作废（反幻觉校验）。

        Args:
            symbol: 标的。
            analytics: 技术分析。
            as_of: 决策日。

        Returns:
            材料 ID → 内容。
        """
        materials: dict[str, str] = {}
        if analytics is not None:
            for index, line in enumerate(analytics.statements(), start=1):
                materials[f"tech{index}"] = line
        for index, item in enumerate(
            self._intel.evidence_for(
                symbol,
                as_of=dt.datetime.combine(as_of, dt.time(15, 0), tzinfo=CST),
                lookback_days=7,
            ),
            start=1,
        ):
            materials[f"news{index}"] = item.cite()
        if not materials:
            materials["empty"] = f"{as_of} 无可用材料"
        return materials

    def _analytics(
        self,
        history: Mapping[Symbol, list[Bar]],
        universe: Sequence[Symbol],
        as_of: TradeDate,
    ) -> dict[Symbol, PositionAnalytics]:
        """构建各标的的持仓与技术分析（支柱②）。

        Args:
            history: 历史行情。
            universe: 候选池。
            as_of: 决策日。

        Returns:
            标的 → 技术分析。
        """
        out: dict[Symbol, PositionAnalytics] = {}
        for symbol in universe:
            bars = history.get(symbol, [])
            if not bars:
                continue
            out[symbol] = build_analytics(
                symbol=symbol,
                as_of=as_of,
                closes=[float(b.close) for b in bars],
                highs=[float(b.high) for b in bars],
                lows=[float(b.low) for b in bars],
                volumes=[float(b.volume) for b in bars],
                ledger=self._ledger,
            )
        return out

    @staticmethod
    def _evidence(signals: Mapping[str, Sequence[Signal]]) -> dict[Symbol, list[Evidence]]:
        """收集各标的的量化依据（支柱①）。

        Args:
            signals: 各策略的信号。

        Returns:
            标的 → 证据列表。
        """
        out: dict[Symbol, list[Evidence]] = {}
        for batch in signals.values():
            for signal in batch:
                out.setdefault(signal.symbol, []).extend(signal.evidence)
        return out

    @staticmethod
    def _counter_evidence(
        signals: Mapping[str, Sequence[Signal]], scores: Mapping[Symbol, float]
    ) -> dict[Symbol, list[Evidence]]:
        """构造反面证据（支柱④，强制项）。

        **没有反面证据的建议视为不完整**，会被 PlanBuilder 剔除。
        这里把"其它策略给出的相反信号"与"打分处于低位"作为反面证据——
        找不到反面证据本身就是一个该被看见的信号。

        Args:
            signals: 各策略的信号。
            scores: 融合打分。

        Returns:
            标的 → 反面证据。
        """
        out: dict[Symbol, list[Evidence]] = {}
        for symbol, score in scores.items():
            notes: list[Evidence] = []
            if score < 0.5:  # noqa: PLR2004 - 0.5 是横截面分位的中点
                notes.append(
                    Evidence(
                        factor="composite_score",
                        value=score,
                        rank_pct=score,
                        contribution=0.0,
                        statement=f"综合打分 {score:.2f} 低于横截面中位，看多依据偏弱",
                    )
                )
            for name, batch in signals.items():
                for signal in batch:
                    if signal.symbol == symbol and signal.score < 0.4:  # noqa: PLR2004
                        notes.append(
                            Evidence(
                                factor=f"{name}_dissent",
                                value=signal.score,
                                rank_pct=signal.score,
                                contribution=0.0,
                                statement=f"策略 {name} 对该标的打分仅 {signal.score:.2f}",
                            )
                        )
            if notes:
                out[symbol] = notes
        return out

    def _intel_evidence(
        self, universe: Sequence[Symbol], *, as_of: TradeDate
    ) -> dict[Symbol, list[IntelEvidence]]:
        """收集情报证据（支柱③）。

        Args:
            universe: 候选池。
            as_of: 决策日。

        Returns:
            标的 → 情报证据。每条都带原文链接与发布时间（红线 I-R4）。
        """
        # 用决策日收盘时刻做 PIT 截断——传 None 会退化成"现在"，
        # 在回测里就等于让当天的建议看到了之后几个月的新闻（红线 I-R5）
        as_of_moment = dt.datetime.combine(as_of, dt.time(15, 0), tzinfo=CST)
        out: dict[Symbol, list[IntelEvidence]] = {}
        for symbol in universe:
            items = self._intel.evidence_for(symbol, as_of=as_of_moment, lookback_days=7)
            hits = [
                IntelEvidence(
                    title=item.title,
                    source=item.source,
                    published_at=item.publish_at,
                    url=item.url,
                    domain=item.domain.value,
                    sentiment=item.sentiment,
                    importance=item.importance,
                    impact=_impact_of(item.sentiment),
                    summary=item.body[:120],
                )
                for item in items
                if item.url
            ]
            if hits:
                out[symbol] = hits
        return out

    @staticmethod
    def _latest_prices(
        history: Mapping[Symbol, list[Bar]], universe: Sequence[Symbol]
    ) -> dict[Symbol, Money]:
        """取各标的的最新收盘价。

        Args:
            history: 历史行情。
            universe: 候选池。

        Returns:
            标的 → 价格。
        """
        return {s: history[s][-1].close for s in universe if history.get(s)}

    def _account_value(self, prices: Mapping[Symbol, Money]) -> Money:
        """账户总资产。

        Args:
            prices: 最新价。

        Returns:
            总资产。无账本时用配置的初始资金——冷启动的人也该能看到建议。
        """
        if self._ledger is None:
            return DEFAULT_COLD_START_CAPITAL
        state = self._ledger.state()
        holdings = sum(
            (prices.get(s, Decimal(0)) * p.qty for s, p in state.positions.items()),
            start=Decimal(0),
        )
        return state.cash + holdings

    def _account_cash(self, total_value: Money) -> Money:
        """可用资金。

        Args:
            total_value: 总资产。

        Returns:
            可用资金。无账本时视作全部为现金。
        """
        return self._ledger.cash if self._ledger is not None else total_value

    @staticmethod
    def _fingerprint(history: Mapping[Symbol, list[Bar]]) -> str:
        """数据指纹（红线 R6）。

        用最后一根 bar 的收盘价与日期做哈希——同一份输入必得同一个指纹，
        换了数据就换指纹，回溯时能立刻发现"输入变了"。

        Args:
            history: 历史行情。

        Returns:
            十六进制指纹。
        """
        import hashlib  # noqa: PLC0415 - 仅此处需要

        material = "|".join(
            f"{symbol}:{bars[-1].trade_date}:{bars[-1].close}:{len(bars)}"
            for symbol, bars in sorted(history.items())
            if bars
        )
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def _strategy_versions(self) -> dict[str, str]:
        """各策略版本（红线 R6）。

        Returns:
            策略 ID → 版本。
        """
        return {
            "momentum_trend": MomentumTrendStrategy().version,
            "etf_rotation": EtfRotationStrategy().version,
        }

    def _param_snapshot(self) -> dict[str, object]:
        """进入 param_hash 的参数快照。

        **含 LLM 的提示词版本与模型 ID**——改提示词等同于改策略。

        Returns:
            参数字典。
        """
        return {
            "portfolio": self._settings.config.portfolio.model_dump(mode="json"),
            "advisor": self._settings.config.advisor.model_dump(mode="json"),
            **self._llm.param_hash_parts(),
        }


def _impact_of(sentiment: float) -> IntelImpact:
    """由情绪分推断情报对建议的作用方向。

    Args:
        sentiment: 情绪分。

    Returns:
        作用方向。中间地带取 NEUTRAL——微弱的情绪倾向不该被说成"支持"。
    """
    threshold = 0.2
    if sentiment > threshold:
        return IntelImpact.SUPPORT
    if sentiment < -threshold:
        return IntelImpact.WEAKEN
    return IntelImpact.NEUTRAL


def _constraints_from(settings: Settings) -> PortfolioConstraints:
    """由配置构造组合约束。

    ``PortfolioConfig`` 与 ``PortfolioConstraints`` 是两套字段——前者面向界面
    （可编辑项），后者面向算法。这里做一次显式映射而不是让两者耦合成同一个类：
    界面加一个展示项不该逼着算法层改签名。

    Args:
        settings: 运行期配置。

    Returns:
        组合约束。
    """
    config = settings.config.portfolio
    risk = settings.config.risk
    return PortfolioConstraints(
        max_holdings=config.top_n,
        rebalance_band=Decimal(str(config.rebalance_band)),
        max_single_position=Decimal(str(getattr(risk, "max_single_position", "0.15"))),
    )
