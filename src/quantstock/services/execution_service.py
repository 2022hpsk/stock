"""执行服务：把计划、通道、风控硬闸与人工确认串起来。

CLI 与界面共用同一实现（见 docs/09-可视化界面规格.md P3 执行页）——
界面绝不能成为风控后门，两条入口必须走同一条代码路径。

**确认是两段式的**：先 ``preview`` 拿到逐单的漂移复核与摘要，
人做完决定后再 ``execute`` 提交决定集合。这样界面可以先整屏展示、
再一次性提交，而不必把交互逻辑塞进执行器里。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from quantstock.advisor.store import PlanStore
from quantstock.advisor.types import TradeIntent, TradePlan
from quantstock.config.settings import Settings
from quantstock.costs import CostModel
from quantstock.execution.brokers import (
    Broker,
    FileBridgeBroker,
    ManualBroker,
    PaperBroker,
)
from quantstock.execution.executor import (
    ConfirmationDecision,
    ExecutionRequest,
    Executor,
)
from quantstock.execution.types import DriftCheck, ExecutionReport, OrderBook, SkipReason
from quantstock.infra.errors import ConfigError
from quantstock.infra.logging import get_logger
from quantstock.infra.types import Money, Side, Symbol
from quantstock.risk.halt import HaltSwitch, HardLimitGuard

# CLI 与界面是"薄"客户端，只允许依赖 services（F20.1 分层契约）。
# 执行相关的契约类型在这里转出，客户端不必、也不允许直接 import execution 层。
__all__ = [
    "ConfirmationDecision",
    "ExecutionPreview",
    "ExecutionReport",
    "ExecutionService",
    "IntentPreview",
    "SkipReason",
]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IntentPreview:
    """单条意图的执行前视图。"""

    intent_id: str
    symbol: Symbol
    side: Side
    qty: int
    price_low: Money
    price_high: Money
    limit_price: Money
    """实际会挂出的限价：买入取区间上沿、卖出取下沿。"""
    estimated_amount: Money
    urgency: str
    verdict: str
    drift: DriftCheck | None
    needs_review: bool
    """漂移超阈值，需要人重新判断而非照单执行。"""


@dataclass(frozen=True, slots=True)
class ExecutionPreview:
    """整个计划的执行前视图。"""

    plan_id: str
    trade_date: str
    broker: str
    requires_live_flag: bool
    halted: bool
    halt_reason: str
    items: tuple[IntentPreview, ...]
    total_buy: Money
    total_sell: Money

    @property
    def review_count(self) -> int:
        """需要重新判断的条数。"""
        return sum(1 for i in self.items if i.needs_review)


class ExecutionService:
    """执行服务。"""

    def __init__(self, settings: Settings, *, order_book: OrderBook | None = None) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
            order_book: 订单簿，进程内保证幂等。跨进程幂等由计划文件的
                ``executed.log`` 保证（见 ``scripts/plan_executor.py``）。
        """
        self._settings = settings
        self._halt = HaltSwitch(settings.var_dir)
        self._broker = self._build_broker(settings)
        self._store = PlanStore(settings.var_dir / "plans")
        self._executor = Executor(
            broker=self._broker,
            halt_switch=self._halt,
            hard_limits=HardLimitGuard(settings.config.risk.hard_limits),
            order_book=order_book,
        )

    @property
    def store(self) -> PlanStore:
        """计划仓库。"""
        return self._store

    @property
    def broker_name(self) -> str:
        """当前通道名。"""
        return self._broker.name

    @staticmethod
    def _build_broker(settings: Settings) -> Broker:
        """按配置构造交易通道。

        Args:
            settings: 运行期配置。

        Returns:
            通道实例。

        Raises:
            ConfigError: 配置了尚未接入的真实通道。
        """
        choice = settings.config.execution.broker
        if choice == "paper":
            return PaperBroker(cost_model=CostModel())
        if choice == "manual":
            return ManualBroker()
        if choice == "file_bridge":
            return FileBridgeBroker(settings.var_dir / "bridge")
        # qmt / ptrade 走执行端分离方案（F15），主系统侧仍是 file_bridge 契约。
        msg = (
            f"通道 {choice} 需要券商程序化权限，当前未接入。"
            "miniQMT 自 2026-07-06 停止新开通，请改用 manual 或 file_bridge。"
        )
        raise ConfigError(msg, broker=choice)

    def preview(self, plan: TradePlan, current_prices: dict[Symbol, Money]) -> ExecutionPreview:
        """生成执行前视图。

        Args:
            plan: 交易计划。
            current_prices: 当前实时价，缺失的标的不做漂移复核。

        Returns:
            执行前视图。
        """
        halt_state = self._halt.state()
        items = tuple(self._preview_intent(intent, current_prices) for intent in plan.intents)
        return ExecutionPreview(
            plan_id=plan.plan_id,
            trade_date=plan.trade_date.isoformat(),
            broker=self._broker.name,
            requires_live_flag=self._broker.requires_live_flag,
            halted=halt_state.halted,
            halt_reason=halt_state.reason,
            items=items,
            total_buy=plan.total_buy_amount,
            total_sell=plan.total_sell_amount,
        )

    def _preview_intent(
        self, intent: TradeIntent, current_prices: dict[Symbol, Money]
    ) -> IntentPreview:
        """单条意图的视图。

        Args:
            intent: 交易意图。
            current_prices: 当前实时价。

        Returns:
            视图条目。
        """
        drift = self._executor.check_drift(intent, current_prices)
        limit = intent.price_high if intent.side is Side.BUY else intent.price_low
        return IntentPreview(
            intent_id=str(intent.intent_id),
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            price_low=intent.price_low,
            price_high=intent.price_high,
            limit_price=limit,
            estimated_amount=intent.estimated_amount,
            urgency=intent.urgency.value,
            verdict=intent.rationale.verdict,
            drift=drift,
            needs_review=drift is not None and drift.is_stale,
        )

    def execute(
        self,
        plan: TradePlan,
        *,
        decisions: Sequence[ConfirmationDecision],
        current_prices: dict[Symbol, Money],
        confirmed_by: str,
        only_symbols: frozenset[Symbol] | None = None,
        live: bool = False,
    ) -> ExecutionReport:
        """按人工决定执行计划。

        **未出现在 ``decisions`` 里的意图一律按跳过处理**——
        默认执行会让"没来得及看"变成"下单了"，这个方向的错误不可逆。

        Args:
            plan: 交易计划。
            decisions: 逐单决定。
            current_prices: 当前实时价。
            confirmed_by: 确认人。
            only_symbols: 只执行指定标的。
            live: 是否使用真实资金通道。

        Returns:
            执行报告。
        """
        by_intent = {d.intent_id: d for d in decisions}

        def confirm(intent: TradeIntent, _: DriftCheck | None) -> ConfirmationDecision:
            return by_intent.get(
                str(intent.intent_id),
                ConfirmationDecision(
                    intent_id=str(intent.intent_id),
                    accepted=False,
                    skip_reason=SkipReason.OTHER,
                    skip_note="未提交确认决定，按跳过处理",
                ),
            )

        report = self._executor.execute(
            ExecutionRequest(
                plan=plan,
                current_prices=current_prices,
                confirmed_by=confirmed_by,
                only_symbols=only_symbols,
            ),
            confirm=confirm,
            live=live,
        )
        # 确认发生在执行之前，但记录放在这里：先让执行器把"必须显式 --live"
        # 之类的前置错误抛出来，避免为一次根本没发生的执行留下确认痕迹。
        self._store.mark_confirmed(plan, confirmed_by=confirmed_by)
        _log.info(
            "execution_service_done",
            plan_id=plan.plan_id,
            aborted=report.aborted,
            submitted=len(report.submitted),
            skipped=len(report.skipped),
        )
        return report

    def cancel_all(self) -> int:
        """撤销所有未成交委托。

        Returns:
            撤单数量。
        """
        return self._executor.cancel_all()

    def bridge_dir(self) -> Path | None:
        """文件桥接目录。

        Returns:
            目录路径；非 file_bridge 通道时 None。
        """
        if isinstance(self._broker, FileBridgeBroker):
            return self._settings.var_dir / "bridge"
        return None

    @staticmethod
    def net_cash_delta(preview: ExecutionPreview) -> Decimal:
        """计划执行后的现金净变化（正为流入）。

        Args:
            preview: 执行前视图。

        Returns:
            现金净变化，未计费用。
        """
        return preview.total_sell - preview.total_buy
