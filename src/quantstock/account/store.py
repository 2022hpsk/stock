"""交易流水的只追加落盘（红线 R8）。

**流水是账本唯一的真相来源**。持仓、批次、快照全部由重放得出，
从不就地修改——所以这个存储只提供"追加"和"读取"，没有 update 也没有 delete。

写错了怎么办？用一笔反向的 ``ADJUST`` 冲正，并写明理由。这比允许改历史
好得多：改历史会让"上周的持仓截图"和"今天重放出来的上周持仓"对不上，
而对不上的时候你根本不知道该信哪个。

格式选 JSONL 而不是 Parquet：流水的量级是每天几笔，人可以直接用编辑器
打开核对，出问题时这一点极其值钱。金额一律存字符串，浮点数往返会
把 ``123.45`` 变成 ``123.44999999999999``（红线 R1）。
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from quantstock.account.types import Transaction, TxnSource, TxnType
from quantstock.infra.clock import now
from quantstock.infra.errors import LedgerError
from quantstock.infra.logging import get_logger
from quantstock.infra.money import money
from quantstock.infra.types import AccountId, IntentId, PlanId, Symbol

__all__ = [
    "TransactionStore",
    "cash_transaction",
    "transaction_from_json",
    "transaction_to_json",
]

_log = get_logger(__name__)

_MONEY_FIELDS = (
    "price",
    "amount",
    "commission",
    "stamp_tax",
    "transfer_fee",
    "exchange_fee",
    "regulatory_fee",
    "dividend_tax",
    "net_cash",
    "ratio",
)


def transaction_to_json(txn: Transaction) -> dict[str, Any]:
    """流水转 JSON。

    Args:
        txn: 流水。

    Returns:
        可落盘的字典。金额全部是字符串。
    """
    payload: dict[str, Any] = {
        "txn_id": txn.txn_id,
        "account_id": str(txn.account_id),
        "txn_type": txn.txn_type.value,
        "trade_date": txn.trade_date.isoformat(),
        "occurred_at": txn.occurred_at.isoformat(),
        "symbol": str(txn.symbol) if txn.symbol else None,
        "qty": txn.qty,
        "source": txn.source.value,
        "plan_id": str(txn.plan_id) if txn.plan_id else None,
        "intent_id": str(txn.intent_id) if txn.intent_id else None,
        "order_id": txn.order_id,
        "note": txn.note,
    }
    payload.update({field: str(getattr(txn, field)) for field in _MONEY_FIELDS})
    return payload


def transaction_from_json(payload: dict[str, Any]) -> Transaction:
    """从 JSON 恢复流水。

    Args:
        payload: 字典。

    Returns:
        流水。

    Raises:
        LedgerError: 字段缺失或格式非法。
    """
    try:
        amounts = {field: money(str(payload.get(field, "0"))) for field in _MONEY_FIELDS}
        symbol = payload.get("symbol")
        plan_id = payload.get("plan_id")
        intent_id = payload.get("intent_id")
        return Transaction(
            txn_id=str(payload["txn_id"]),
            account_id=AccountId(str(payload["account_id"])),
            txn_type=TxnType(str(payload["txn_type"])),
            trade_date=dt.date.fromisoformat(str(payload["trade_date"])),
            occurred_at=dt.datetime.fromisoformat(str(payload["occurred_at"])),
            symbol=Symbol(str(symbol)) if symbol else None,
            qty=int(payload.get("qty", 0)),
            source=TxnSource(str(payload.get("source", TxnSource.MANUAL.value))),
            plan_id=PlanId(str(plan_id)) if plan_id else None,
            intent_id=IntentId(str(intent_id)) if intent_id else None,
            order_id=payload.get("order_id"),
            note=str(payload.get("note", "")),
            **amounts,
        )
    except (KeyError, ValueError, TypeError) as exc:
        msg = f"流水记录格式非法：{exc}"
        raise LedgerError(msg, payload=str(payload)[:200]) from exc


class TransactionStore:
    """只追加的流水存储。"""

    def __init__(self, path: Path) -> None:
        """初始化。

        Args:
            path: ``transactions.jsonl`` 路径。
        """
        self._path = path

    @property
    def path(self) -> Path:
        """流水文件路径。"""
        return self._path

    def append(self, txn: Transaction) -> None:
        """追加一笔流水。

        **幂等**：``txn_id`` 已存在时直接跳过。执行器重试、界面重复提交
        都不该产生两笔一样的成交。

        Args:
            txn: 流水。
        """
        if txn.txn_id in self._existing_ids():
            _log.info("txn_already_recorded", txn_id=txn.txn_id)
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(transaction_to_json(txn), ensure_ascii=False) + "\n")
        _log.info(
            "txn_recorded",
            txn_id=txn.txn_id,
            type=txn.txn_type.value,
            symbol=str(txn.symbol) if txn.symbol else "-",
        )

    def extend(self, transactions: list[Transaction]) -> int:
        """批量追加。

        Args:
            transactions: 流水列表。

        Returns:
            实际写入的条数（已存在的不计）。
        """
        known = self._existing_ids()
        fresh = [t for t in transactions if t.txn_id not in known]
        if not fresh:
            return 0

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            for txn in fresh:
                handle.write(json.dumps(transaction_to_json(txn), ensure_ascii=False) + "\n")
        return len(fresh)

    def read_all(self) -> list[Transaction]:
        """读取全部流水。

        Returns:
            流水列表，按写入顺序。重放前会自行排序，这里不排。

        Raises:
            LedgerError: 某行格式非法。**不跳过坏行**——账本少一笔就是错的，
                静默跳过会让持仓凭空少掉一部分而没人发现。
        """
        if not self._path.exists():
            return []

        out: list[Transaction] = []
        for lineno, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                msg = f"流水第 {lineno} 行不是合法 JSON"
                raise LedgerError(msg, path=str(self._path), line=lineno) from exc
            out.append(transaction_from_json(payload))
        return out

    def count(self) -> int:
        """流水条数。

        Returns:
            条数。
        """
        if not self._path.exists():
            return 0
        lines = self._path.read_text(encoding="utf-8").splitlines()
        return sum(1 for line in lines if line.strip())

    def next_txn_id(self, prefix: str = "txn") -> str:
        """生成下一个流水 ID。

        用「时间戳 + 序号」而不是纯自增：手工录入与执行器写入可能交错，
        纯自增在并发下会撞号，而撞号会被幂等逻辑当成重复而**静默丢弃**。

        Args:
            prefix: ID 前缀。

        Returns:
            流水 ID。
        """
        return f"{prefix}-{now().strftime('%Y%m%d%H%M%S')}-{self.count() + 1:05d}"

    def _existing_ids(self) -> set[str]:
        """已有的流水 ID 集合。

        Returns:
            ID 集合。
        """
        if not self._path.exists():
            return set()
        ids: set[str] = set()
        for line in self._path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                ids.add(str(json.loads(text)["txn_id"]))
            except (json.JSONDecodeError, KeyError):
                continue  # 坏行留给 read_all 报错，这里只管去重
        return ids


def cash_transaction(
    *,
    txn_id: str,
    account_id: AccountId,
    txn_type: TxnType,
    trade_date: dt.date,
    amount: Decimal,
    note: str = "",
) -> Transaction:
    """构造一笔资金流水（入金 / 出金）。

    ``net_cash`` 的符号由类型决定而不是由调用方传——让调用方自己决定符号，
    迟早会出现"入金记成负数"这种把账户余额直接算错的错误。

    Args:
        txn_id: 流水 ID。
        account_id: 账户。
        txn_type: ``DEPOSIT`` 或 ``WITHDRAW``。
        trade_date: 日期。
        amount: 金额，正数。
        note: 备注。

    Returns:
        流水。

    Raises:
        LedgerError: 类型不是资金类，或金额非正。
    """
    if txn_type not in {TxnType.DEPOSIT, TxnType.WITHDRAW}:
        msg = f"{txn_type.value} 不是资金类流水"
        raise LedgerError(msg, txn_type=txn_type.value)
    if amount <= 0:
        msg = "资金流水金额必须为正，方向由类型决定"
        raise LedgerError(msg, amount=str(amount))

    signed = amount if txn_type is TxnType.DEPOSIT else -amount
    return Transaction(
        txn_id=txn_id,
        account_id=account_id,
        txn_type=txn_type,
        trade_date=trade_date,
        occurred_at=now(),
        amount=amount,
        net_cash=signed,
        source=TxnSource.MANUAL,
        note=note,
    )
