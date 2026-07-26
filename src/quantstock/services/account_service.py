"""账户服务：持仓、批次、资金流水（docs/11-持仓账本规格.md）。

**这是此前缺失的一环**。``AdvisorService`` 一直拿不到账本（``ledger=None``），
于是永远按冷启动出建议——它不知道你已经持有什么，因此：

- 支柱②"持仓与技术分析"退化成只有技术形态，没有真实成本与持有期；
- "再持有 N 天可免红利税"这类提示根本不会出现；
- 组合层的差分调仓拿空持仓去比，卖出建议一条也生不成。

补上流水存储后，这些全部自然恢复——因为它们本来就是由账本推导的。

红线 R8：流水只追加不修改，持仓完全由重放得出。所以这里没有"修改持仓"的
方法，只有"记一笔流水"。写错了用反向 ``ADJUST`` 冲正。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from quantstock.account.ledger import Ledger, replay
from quantstock.account.store import TransactionStore, cash_transaction
from quantstock.account.types import LedgerState, Position, Transaction, TxnSource, TxnType
from quantstock.config.settings import Settings
from quantstock.infra.clock import now, today
from quantstock.infra.errors import LedgerError
from quantstock.infra.logging import get_logger
from quantstock.infra.types import AccountId, Money, Symbol, TradeDate

# 界面与 CLI 是"薄"客户端，只允许依赖 services（F20.1 分层契约）。
# 账本相关的契约类型在这里转出。
__all__ = [
    "AccountService",
    "AccountSummary",
    "LedgerState",
    "Position",
    "Transaction",
    "TxnType",
]

_log = get_logger(__name__)

DEFAULT_ACCOUNT_ID = AccountId("main")


@dataclass(frozen=True, slots=True)
class AccountSummary:
    """账户总览，供仪表盘与账户页。"""

    account_id: str
    as_of: TradeDate
    cash: Money
    market_value: Money
    total_value: Money
    position_count: int
    realized_pnl: Money
    unrealized_pnl: Money
    total_fee: Money
    total_dividend: Money
    total_dividend_tax: Money
    total_deposit: Money
    total_withdraw: Money
    transactions: int
    priced_symbols: int
    unpriced_symbols: tuple[str, ...]
    """没拿到现价的持仓。**必须显式列出**——它们按 0 计入市值，
    静默的话总资产会莫名其妙地少一截，而用户会以为是自己算错了。"""

    @property
    def is_empty(self) -> bool:
        """是否是空账户（冷启动）。"""
        return self.transactions == 0

    @property
    def message(self) -> str:
        """一行摘要。

        Returns:
            摘要文本。
        """
        if self.is_empty:
            return "账本为空。录入持仓或入金后，建议才能基于真实成本与持有期给出"
        base = (
            f"总资产 {self.total_value}（现金 {self.cash} + 持仓 {self.market_value}），"
            f"{self.position_count} 只持仓，浮动盈亏 {self.unrealized_pnl}"
        )
        if self.unpriced_symbols:
            base += f"；{len(self.unpriced_symbols)} 只标的缺现价，已按 0 计"
        return base


class AccountService:
    """账户与持仓编排。"""

    def __init__(self, settings: Settings, *, account_id: AccountId | None = None) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
            account_id: 账户标识。
        """
        self._settings = settings
        self._account_id = account_id or DEFAULT_ACCOUNT_ID
        self._store = TransactionStore(
            settings.var_dir / "account" / f"{self._account_id}.transactions.jsonl"
        )

    @property
    def account_id(self) -> AccountId:
        """账户标识。"""
        return self._account_id

    @property
    def store(self) -> TransactionStore:
        """流水存储。"""
        return self._store

    def ledger(self, *, as_of: TradeDate | None = None) -> Ledger | None:
        """重放出账本。

        Args:
            as_of: 只重放到该日期（含）；None 表示全部。

        Returns:
            账本；没有任何流水时 None——冷启动是合法状态，
            不该用一个假的空账本冒充"账户里什么都没有"。
        """
        transactions = self._store.read_all()
        if not transactions:
            return None
        return replay(self._account_id, transactions, as_of=as_of)

    def state(self, *, as_of: TradeDate | None = None) -> LedgerState | None:
        """账本状态快照。

        Args:
            as_of: 时点。

        Returns:
            状态；空账本时 None。
        """
        ledger = self.ledger(as_of=as_of)
        return None if ledger is None else ledger.state(as_of=as_of)

    def positions(self, *, as_of: TradeDate | None = None) -> dict[Symbol, Position]:
        """当前持仓。

        Args:
            as_of: 时点。

        Returns:
            标的 → 持仓。空账本时空字典。
        """
        ledger = self.ledger(as_of=as_of)
        return {} if ledger is None else ledger.positions(as_of=as_of)

    def summary(
        self,
        prices: dict[Symbol, Money] | None = None,
        *,
        as_of: TradeDate | None = None,
    ) -> AccountSummary:
        """账户总览。

        Args:
            prices: 各标的现价。缺失的标的按 0 计，并在 ``unpriced_symbols``
                里列出——静默按 0 会让总资产莫名少一截。
            as_of: 时点。

        Returns:
            总览。
        """
        moment = as_of or today()
        quotes = prices or {}
        state = self.state(as_of=moment)
        count = self._store.count()

        if state is None:
            zero = Decimal(0)
            return AccountSummary(
                account_id=str(self._account_id),
                as_of=moment,
                cash=zero,
                market_value=zero,
                total_value=zero,
                position_count=0,
                realized_pnl=zero,
                unrealized_pnl=zero,
                total_fee=zero,
                total_dividend=zero,
                total_dividend_tax=zero,
                total_deposit=zero,
                total_withdraw=zero,
                transactions=count,
                priced_symbols=0,
                unpriced_symbols=(),
            )

        missing = tuple(str(s) for s in state.positions if s not in quotes)
        market_value = state.holdings_value(quotes)
        unrealized = sum(
            (
                quotes.get(sym, Decimal(0)) * pos.qty - pos.cost_basis_avg * pos.qty
                for sym, pos in state.positions.items()
                if sym in quotes
            ),
            start=Decimal(0),
        )

        return AccountSummary(
            account_id=str(state.account_id),
            as_of=state.as_of,
            cash=state.cash,
            market_value=market_value,
            total_value=state.cash + market_value,
            position_count=len(state.positions),
            realized_pnl=state.realized_pnl,
            unrealized_pnl=unrealized,
            total_fee=state.total_fee,
            total_dividend=state.total_dividend,
            total_dividend_tax=state.total_dividend_tax,
            total_deposit=state.total_deposit,
            total_withdraw=state.total_withdraw,
            transactions=count,
            priced_symbols=len(state.positions) - len(missing),
            unpriced_symbols=missing,
        )

    def record(self, txn: Transaction) -> Transaction:
        """记一笔流水。

        **写入前先试重放**：一笔非法流水（卖出超过持仓、日期倒流）如果落了盘，
        之后每次重放都会抛错，账本就彻底打不开了。先在内存里验一遍，
        坏数据根本进不去。

        Args:
            txn: 流水。

        Returns:
            已记录的流水。

        Raises:
            LedgerError: 流水非法。
        """
        existing = self._store.read_all()
        replay(self._account_id, [*existing, txn])  # 非法则在此抛出，不会落盘
        self._store.append(txn)
        return txn

    def deposit(
        self, amount: Money, *, trade_date: TradeDate | None = None, note: str = ""
    ) -> Transaction:
        """入金。

        Args:
            amount: 金额，正数。
            trade_date: 日期。
            note: 备注。

        Returns:
            流水。
        """
        return self.record(
            cash_transaction(
                txn_id=self._store.next_txn_id("cash"),
                account_id=self._account_id,
                txn_type=TxnType.DEPOSIT,
                trade_date=trade_date or today(),
                amount=amount,
                note=note,
            )
        )

    def withdraw(
        self, amount: Money, *, trade_date: TradeDate | None = None, note: str = ""
    ) -> Transaction:
        """出金。

        Args:
            amount: 金额，正数。方向由类型决定。
            trade_date: 日期。
            note: 备注。

        Returns:
            流水。
        """
        return self.record(
            cash_transaction(
                txn_id=self._store.next_txn_id("cash"),
                account_id=self._account_id,
                txn_type=TxnType.WITHDRAW,
                trade_date=trade_date or today(),
                amount=amount,
                note=note,
            )
        )

    def trade(
        self,
        *,
        symbol: Symbol,
        side: str,
        qty: int,
        price: Money,
        trade_date: TradeDate | None = None,
        commission: Money = Decimal(0),
        stamp_tax: Money = Decimal(0),
        transfer_fee: Money = Decimal(0),
        note: str = "",
        source: TxnSource = TxnSource.MANUAL,
    ) -> Transaction:
        """记一笔成交。

        ``qty`` 与 ``net_cash`` 的符号由 ``side`` 决定而不是由调用方传——
        让调用方自己决定符号，迟早会出现"卖出记成正数"这种把持仓算成两倍的错误。

        Args:
            symbol: 标的。
            side: ``buy`` 或 ``sell``。
            qty: 数量，正数。
            price: 成交价。
            trade_date: 成交日。
            commission: 佣金。
            stamp_tax: 印花税（仅卖出）。
            transfer_fee: 过户费。
            note: 备注。
            source: 流水来源，用于追溯（红线 R6）。

        Returns:
            流水。

        Raises:
            LedgerError: 方向非法或数量非正。
        """
        normalized = side.strip().lower()
        if normalized not in {"buy", "sell"}:
            msg = f"成交方向必须是 buy 或 sell，收到 {side!r}"
            raise LedgerError(msg, side=side)
        if qty <= 0:
            msg = "成交数量必须为正，方向由 side 决定"
            raise LedgerError(msg, qty=str(qty))

        is_buy = normalized == "buy"
        gross = price * qty
        fees = commission + stamp_tax + transfer_fee
        # 买入：付出货款并承担费用；卖出：收到货款后扣除费用
        net_cash = -(gross + fees) if is_buy else gross - fees

        return self.record(
            Transaction(
                txn_id=self._store.next_txn_id("trade"),
                account_id=self._account_id,
                txn_type=TxnType.BUY if is_buy else TxnType.SELL,
                trade_date=trade_date or today(),
                occurred_at=now(),
                symbol=symbol,
                qty=qty if is_buy else -qty,
                price=price,
                amount=gross,
                commission=commission,
                stamp_tax=stamp_tax,
                transfer_fee=transfer_fee,
                net_cash=net_cash,
                source=source,
                note=note,
            )
        )

    def transactions(self, *, limit: int | None = None) -> list[Transaction]:
        """流水明细。

        Args:
            limit: 最多返回条数，取最近的。

        Returns:
            流水列表，按时间倒序。
        """
        ordered = sorted(
            self._store.read_all(),
            key=lambda t: (t.trade_date, t.occurred_at, t.txn_id),
            reverse=True,
        )
        return ordered[:limit] if limit else ordered

    def tax_countdown(self, *, as_of: TradeDate | None = None) -> dict[Symbol, int]:
        """各持仓距满一年免红利税的天数。

        高股息标的差几天卖掉，红利税从免征跳到 10%，可能值几千元。
        这是**只有批次级账本才算得出来**的东西——平均成本里没有建仓日。

        Args:
            as_of: 时点。

        Returns:
            标的 → 剩余天数。已满一年或算不出的不列入。
        """
        ledger = self.ledger(as_of=as_of)
        if ledger is None:
            return {}
        moment = as_of or today()
        out: dict[Symbol, int] = {}
        for symbol in ledger.positions(as_of=moment):
            days = ledger.days_to_tax_free(symbol, as_of=moment)
            if days is not None and days > 0:
                out[symbol] = days
        return out

    def import_transactions(self, rows: Sequence[dict[str, object]]) -> int:
        """批量导入流水（券商对账单 / CSV）。

        Args:
            rows: 行数据，字段同 ``trade`` 的参数。

        Returns:
            成功导入的条数。

        Raises:
            LedgerError: 任一行非法。**整批失败而不是部分成功**——
                导入一半的对账单比完全没导入更难收拾。
        """
        parsed: list[Transaction] = []
        for index, row in enumerate(rows, 1):
            try:
                parsed.append(self._row_to_transaction(row, index))
            except (KeyError, ValueError, TypeError) as exc:
                msg = f"第 {index} 行无法解析：{exc}"
                raise LedgerError(msg, row=str(row)[:200]) from exc

        existing = self._store.read_all()
        replay(self._account_id, [*existing, *parsed])  # 整批先验，坏数据不落盘
        written = self._store.extend(parsed)
        _log.info("transactions_imported", rows=len(rows), written=written)
        return written

    def _row_to_transaction(self, row: dict[str, object], index: int) -> Transaction:
        """把一行导入数据转成流水。

        Args:
            row: 行数据。
            index: 行号，用于生成 ID。

        Returns:
            流水。
        """
        side = str(row.get("side", "buy")).strip().lower()
        is_buy = side == "buy"
        qty = int(str(row["qty"]))
        price = Decimal(str(row["price"]))
        commission = Decimal(str(row.get("commission", "0")))
        stamp_tax = Decimal(str(row.get("stamp_tax", "0")))
        transfer_fee = Decimal(str(row.get("transfer_fee", "0")))
        gross = price * qty
        fees = commission + stamp_tax + transfer_fee

        return Transaction(
            txn_id=str(row.get("txn_id") or f"{self._store.next_txn_id('import')}-{index}"),
            account_id=self._account_id,
            txn_type=TxnType.BUY if is_buy else TxnType.SELL,
            trade_date=dt.date.fromisoformat(str(row["trade_date"])),
            occurred_at=now(),
            symbol=Symbol(str(row["symbol"])),
            qty=qty if is_buy else -qty,
            price=price,
            amount=gross,
            commission=commission,
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            net_cash=-(gross + fees) if is_buy else gross - fees,
            source=TxnSource.BROKER_SYNC,
            note=str(row.get("note", "")),
        )
