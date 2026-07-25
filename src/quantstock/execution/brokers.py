"""券商通道实现。

规范见 docs/03-功能规格.md F8.2、F15。

**背景**：miniQMT 自 2026-07-06 停止新开通，券商程序化接口不再是可依赖的前置条件。
因此这里的三个通道都**不需要任何券商权限**：

- ``PaperBroker``：本地模拟撮合，默认通道，用于全链路演练与模拟盘验证期；
- ``ManualBroker``：输出可复制的手工执行清单，你在券商 App 里下单；
- ``FileBridgeBroker``：写出计划文件给执行端，读回成交回报（执行端分离，见 F15）。

QMT/PTrade 通道复用 ``FileBridgeBroker`` 的文件契约——
执行端只需一个 300 行的脚本，换券商时主系统零改动。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Protocol, runtime_checkable

from quantstock.costs import CostModel
from quantstock.execution.types import (
    BrokerOrder,
    OrderStatus,
    TradeFill,
)
from quantstock.infra.clock import now
from quantstock.infra.errors import BrokerConnectionError, OrderRejectedError
from quantstock.infra.logging import get_logger
from quantstock.infra.money import ZERO, quantize_cny
from quantstock.infra.types import Money, Side

__all__ = [
    "Broker",
    "FileBridgeBroker",
    "ManualBroker",
    "PaperBroker",
]

_log = get_logger(__name__)


@runtime_checkable
class Broker(Protocol):
    """券商通道。

    所有实现通过同一套集成测试，以 ``PaperBroker`` 为行为基线——
    行为不一致的通道会让"模拟盘跑通了"变成一句空话。
    """

    name: str
    requires_live_flag: bool
    """是否为真实资金通道。真实通道需显式 ``--live``（红线 R5）。"""

    def submit(self, orders: Sequence[BrokerOrder]) -> list[BrokerOrder]:
        """提交订单。

        Args:
            orders: 已确认的订单。

        Returns:
            带提交结果的订单。

        Raises:
            BrokerConnectionError: 通道不可用。
            OrderRejectedError: 券商拒绝。
        """
        ...

    def fetch_fills(self, order_ids: Sequence[str]) -> list[TradeFill]:
        """拉取成交回报。

        Args:
            order_ids: 订单号。

        Returns:
            成交列表。
        """
        ...

    def cancel_all(self) -> int:
        """撤销所有未成交委托。

        Returns:
            撤单数量。
        """
        ...


class PaperBroker:
    """本地模拟撮合。

    **默认通道**。撮合假设刻意保守：按委托价全额成交并扣除完整费用。
    这不是为了贴近真实（真实成交会有滑点与部分成交），而是为了让
    "模拟盘赚钱、实盘亏钱"这类落差尽可能小——模拟乐观一分，实盘失望十分。
    """

    name = "paper"
    requires_live_flag = False

    def __init__(self, *, cost_model: CostModel | None = None) -> None:
        """初始化。

        Args:
            cost_model: 成本模型。与回测、实盘共用同一实现。
        """
        self._costs = cost_model or CostModel()
        self._fills: dict[str, TradeFill] = {}
        self._live_orders: dict[str, BrokerOrder] = {}

    def submit(self, orders: Sequence[BrokerOrder]) -> list[BrokerOrder]:
        """模拟撮合。

        Args:
            orders: 已确认的订单。

        Returns:
            全部标记为已成交的订单。

        Raises:
            OrderRejectedError: 订单未经确认。
        """
        results: list[BrokerOrder] = []
        moment = now()
        for order in orders:
            if order.status is not OrderStatus.CONFIRMED:
                msg = "只有已确认的订单才能提交（红线 R5）"
                raise OrderRejectedError(msg, order_id=order.order_id, status=order.status.value)

            fee = self._costs.compute(
                amount=order.amount, side=order.side, trade_date=moment.date()
            ).total
            fill = TradeFill(
                fill_id=uuid.uuid4().hex[:12],
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=order.price,
                filled_at=moment,
                fee=fee,
            )
            self._fills[order.order_id] = fill

            submitted = order.with_status(
                OrderStatus.SUBMITTED, submitted_at=moment, broker_order_id=f"paper-{fill.fill_id}"
            )
            results.append(
                submitted.with_status(
                    OrderStatus.FILLED, filled_qty=order.qty, avg_fill_price=order.price
                )
            )
        _log.info("paper_orders_filled", count=len(results))
        return results

    def fetch_fills(self, order_ids: Sequence[str]) -> list[TradeFill]:
        """取成交回报。

        Args:
            order_ids: 订单号。

        Returns:
            成交列表。
        """
        return [self._fills[oid] for oid in order_ids if oid in self._fills]

    def cancel_all(self) -> int:
        """模拟通道下所有订单立即成交，无未成交委托。

        Returns:
            始终为 0。
        """
        self._live_orders.clear()
        return 0


class ManualBroker:
    """手工执行清单。

    不需要任何券商权限——生成可直接照着在券商 App 里下单的清单，
    成交后由用户回填。这是 miniQMT 停开后最现实的落地方案。
    """

    name = "manual"
    requires_live_flag = False

    def __init__(self) -> None:
        """初始化。"""
        self._pending: dict[str, BrokerOrder] = {}
        self._fills: dict[str, TradeFill] = {}

    def submit(self, orders: Sequence[BrokerOrder]) -> list[BrokerOrder]:
        """生成待手工执行的订单。

        Args:
            orders: 已确认的订单。

        Returns:
            标记为 SUBMITTED 的订单，等待用户回填成交。
        """
        moment = now()
        results: list[BrokerOrder] = []
        for order in orders:
            submitted = order.with_status(OrderStatus.SUBMITTED, submitted_at=moment)
            self._pending[order.order_id] = submitted
            results.append(submitted)
        return results

    @staticmethod
    def build_checklist(orders: Sequence[BrokerOrder]) -> list[str]:
        """生成人类可读的执行清单。

        格式刻意做成可直接照抄的样子——每行包含方向、代码、数量、限价，
        不掺杂解释文字，避免在券商 App 里手忙脚乱时看错。

        Args:
            orders: 待执行订单。

        Returns:
            清单行列表。
        """
        lines: list[str] = []
        for i, order in enumerate(orders, start=1):
            action = "买入" if order.side is Side.BUY else "卖出"
            lines.append(
                f"{i}. [{action}] {order.symbol}  {order.qty} 股  "
                f"限价 {order.price}  约 {quantize_cny(order.amount)} 元"
            )
        return lines

    def record_fill(self, order_id: str, *, qty: int, price: Money, fee: Money = ZERO) -> TradeFill:
        """回填手工成交。

        Args:
            order_id: 订单号。
            qty: 成交数量。
            price: 成交价。
            fee: 费用。

        Returns:
            成交记录。

        Raises:
            OrderRejectedError: 订单不存在或数量非法。
        """
        order = self._pending.get(order_id)
        if order is None:
            msg = "回填的订单不存在"
            raise OrderRejectedError(msg, order_id=order_id)
        if not 0 < qty <= order.qty:
            msg = f"回填数量 {qty} 必须在 (0, {order.qty}] 之间"
            raise OrderRejectedError(msg, order_id=order_id)

        fill = TradeFill(
            fill_id=uuid.uuid4().hex[:12],
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            filled_at=now(),
            fee=fee,
        )
        self._fills[order_id] = fill
        return fill

    def fetch_fills(self, order_ids: Sequence[str]) -> list[TradeFill]:
        """取已回填的成交。

        Args:
            order_ids: 订单号。

        Returns:
            成交列表。
        """
        return [self._fills[oid] for oid in order_ids if oid in self._fills]

    def cancel_all(self) -> int:
        """清空待执行清单。

        Returns:
            清除的订单数。
        """
        count = len(self._pending)
        self._pending.clear()
        return count


class FileBridgeBroker:
    """文件桥接通道（执行端分离，F15）。

    研究端写出 ``plan-<date>.json``，执行端读取并下单，回写 ``fills-<date>.json``。

    这样设计是因为 QMT/PTrade 的策略必须运行在它们各自受限的环境里
    （Windows 客户端内置 Python / 券商机房无外网、不可装三方包），
    把整个 quantstock 塞进去不现实。执行端只需一个零依赖的小脚本。

    **计划文件是唯一契约**：执行端不做任何决策，只执行与回报。
    换券商时只需重写执行器，主系统零改动。
    """

    name = "file_bridge"
    requires_live_flag = True
    SCHEMA_VERSION = 1

    def __init__(self, bridge_dir: Path) -> None:
        """初始化。

        Args:
            bridge_dir: 桥接目录，研究端与执行端共享。
        """
        self._dir = Path(bridge_dir)

    @property
    def plan_path(self) -> Path:
        """当日计划文件路径。"""
        return self._dir / f"plan-{now().date().isoformat()}.json"

    @property
    def fills_path(self) -> Path:
        """当日成交回报文件路径。"""
        return self._dir / f"fills-{now().date().isoformat()}.json"

    def submit(self, orders: Sequence[BrokerOrder]) -> list[BrokerOrder]:
        """写出计划文件。

        Args:
            orders: 已确认的订单。

        Returns:
            标记为 SUBMITTED 的订单。

        Raises:
            BrokerConnectionError: 目录不可写。
        """
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "generated_at": now().isoformat(),
            "orders": [
                {
                    "order_id": o.order_id,
                    "intent_id": str(o.intent_id),
                    "symbol": str(o.symbol),
                    "side": o.side.value,
                    "qty": o.qty,
                    "price": str(o.price),
                    "price_type": o.price_type.value,
                }
                for o in orders
            ],
        }
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self.plan_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            msg = "无法写出计划文件"
            raise BrokerConnectionError(msg, path=str(self.plan_path), error=str(exc)) from exc

        moment = now()
        _log.info("bridge_plan_written", path=str(self.plan_path), orders=len(orders))
        return [o.with_status(OrderStatus.SUBMITTED, submitted_at=moment) for o in orders]

    def fetch_fills(self, order_ids: Sequence[str]) -> list[TradeFill]:
        """读取执行端回写的成交。

        Args:
            order_ids: 订单号。

        Returns:
            成交列表；文件不存在时为空。

        Raises:
            BrokerConnectionError: 回报文件格式非法。
        """
        if not self.fills_path.exists():
            return []
        try:
            payload = json.loads(self.fills_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            msg = "成交回报文件无法解析"
            raise BrokerConnectionError(msg, path=str(self.fills_path), error=str(exc)) from exc

        wanted = set(order_ids)
        moment = now()
        return [
            TradeFill(
                fill_id=str(row.get("fill_id", uuid.uuid4().hex[:12])),
                order_id=str(row["order_id"]),
                symbol=row["symbol"],
                side=Side(row["side"]),
                qty=int(row["qty"]),
                price=Decimal(str(row["price"])),
                filled_at=moment,
                fee=Decimal(str(row.get("fee", "0"))),
            )
            for row in payload.get("fills", [])
            if str(row.get("order_id")) in wanted
        ]

    def cancel_all(self) -> int:
        """写出撤单指令文件。

        Returns:
            始终为 0——实际撤单数由执行端回报。
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "cancel-all.flag").write_text(now().isoformat(), encoding="utf-8")
        return 0
