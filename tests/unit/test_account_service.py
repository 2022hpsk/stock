"""交易流水存储与账户服务测试（红线 R8、docs/11-持仓账本规格.md）。

**这一层补的是一个真实的功能缺口**：在它出现之前，``AdvisorService`` 恒以
``ledger=None`` 运行——系统永远不知道你持有什么，于是支柱②没有真实成本、
"再持有 N 天免红利税"永远不出现、卖出建议一条也生不成。1000 多个测试全绿，
因为没有任何测试问过"账本从哪来"。

所以这里的重点不是"流水能不能存"，而是几条**只有真的用起来才会暴露**的性质：

- 非法流水绝不能落盘。落了之后每次重放都会抛错，账本就彻底打不开了；
- 方向由 ``side`` / 类型决定，不由调用方传符号。让调用方决定符号，
  迟早出现"卖出记成正数"这种把持仓算成两倍的错误；
- 重放必须可重入：同一份流水读两次，得到完全一样的持仓。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

from quantstock.account.store import TransactionStore, transaction_from_json, transaction_to_json
from quantstock.account.types import Transaction, TxnSource, TxnType
from quantstock.config.models import RootConfig
from quantstock.config.settings import Secrets, Settings
from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import LedgerError
from quantstock.infra.money import money
from quantstock.infra.types import AccountId, Symbol
from quantstock.services.account_service import AccountService

MAOTAI = Symbol("600519.SH")
PINGAN = Symbol("601318.SH")
NOW = dt.datetime(2026, 7, 24, 15, 30, tzinfo=CST)
ACCOUNT = AccountId("main")


@pytest.fixture(autouse=True)
def _frozen() -> None:
    """冻结时钟（红线 R3）。"""
    set_clock(FrozenClock(NOW))


@pytest.fixture
def service(tmp_path: Path) -> AccountService:
    config = RootConfig()
    config.app.var_dir = str(tmp_path / "var")
    settings = Settings(config=config, secrets=Secrets(_env_file=None), config_dir=tmp_path / "cfg")
    return AccountService(settings)


def _buy(txn_id: str, *, day: int = 1, qty: int = 100, price: str = "100") -> Transaction:
    """构造一笔买入。

    Args:
        txn_id: 流水 ID。
        day: 2026-07 的日。
        qty: 数量。
        price: 价格。

    Returns:
        流水。
    """
    gross = money(price) * qty
    return Transaction(
        txn_id=txn_id,
        account_id=ACCOUNT,
        txn_type=TxnType.BUY,
        trade_date=dt.date(2026, 7, day),
        occurred_at=dt.datetime(2026, 7, day, 10, 0, tzinfo=CST),
        symbol=MAOTAI,
        qty=qty,
        price=money(price),
        amount=gross,
        net_cash=-gross,
        source=TxnSource.MANUAL,
    )


class TestSerialization:
    """流水的 JSON 往返。"""

    def test_roundtrip_preserves_everything(self) -> None:
        original = _buy("t1")
        restored = transaction_from_json(transaction_to_json(original))
        assert restored == original

    def test_amounts_are_stored_as_strings(self) -> None:
        # 浮点往返会把 123.45 变成 123.44999999999999。存的是钱，不能这样（红线 R1）
        payload = transaction_to_json(_buy("t1", price="123.45"))
        assert payload["price"] == "123.45"
        assert isinstance(payload["price"], str)

    def test_precision_survives_a_roundtrip(self) -> None:
        original = _buy("t1", price="1596.523", qty=137)
        restored = transaction_from_json(transaction_to_json(original))
        assert restored.price == Decimal("1596.523")
        assert restored.net_cash == original.net_cash

    def test_malformed_payload_raises(self) -> None:
        with pytest.raises(LedgerError, match="格式非法"):
            transaction_from_json({"txn_id": "x"})


class TestTransactionStore:
    """只追加存储。"""

    def test_append_then_read(self, tmp_path: Path) -> None:
        store = TransactionStore(tmp_path / "txn.jsonl")
        store.append(_buy("t1"))
        assert [t.txn_id for t in store.read_all()] == ["t1"]

    def test_append_is_idempotent(self, tmp_path: Path) -> None:
        # 执行器重试、界面重复提交都不该产生两笔一样的成交
        store = TransactionStore(tmp_path / "txn.jsonl")
        store.append(_buy("t1"))
        store.append(_buy("t1"))
        assert store.count() == 1

    def test_extend_skips_known_ids(self, tmp_path: Path) -> None:
        store = TransactionStore(tmp_path / "txn.jsonl")
        store.append(_buy("t1"))
        written = store.extend([_buy("t1"), _buy("t2", day=2)])
        assert written == 1
        assert store.count() == 2

    def test_missing_file_reads_empty(self, tmp_path: Path) -> None:
        assert TransactionStore(tmp_path / "nope.jsonl").read_all() == []

    def test_corrupt_line_raises_instead_of_being_skipped(self, tmp_path: Path) -> None:
        # 账本少一笔就是错的。静默跳过坏行会让持仓凭空少掉一部分而没人发现
        path = tmp_path / "txn.jsonl"
        store = TransactionStore(path)
        store.append(_buy("t1"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{ not json\n")

        with pytest.raises(LedgerError, match="不是合法 JSON"):
            store.read_all()

    def test_next_id_is_unique(self, tmp_path: Path) -> None:
        store = TransactionStore(tmp_path / "txn.jsonl")
        first = store.next_txn_id()
        store.append(_buy(first))
        assert store.next_txn_id() != first


class TestAccountService:
    """账户服务。"""

    def test_empty_account_is_a_valid_state(self, service: AccountService) -> None:
        # 冷启动是合法状态，不该用一个假的空账本冒充"账户里什么都没有"
        assert service.ledger() is None
        assert service.positions() == {}
        assert service.summary().is_empty is True

    def test_deposit_then_buy(self, service: AccountService) -> None:
        service.deposit(money("100000"))
        service.trade(symbol=MAOTAI, side="buy", qty=100, price=money("1000"))

        state = service.state()
        assert state is not None
        assert state.cash == money("0")  # 100000 入金 - 100 股 × 1000
        assert state.positions[MAOTAI].qty == 100

    def test_buy_direction_is_derived_not_supplied(self, service: AccountService) -> None:
        # 让调用方自己传符号，迟早会出现"卖出记成正数"这种把持仓算成两倍的错误
        service.deposit(money("100000"))
        txn = service.trade(symbol=MAOTAI, side="buy", qty=100, price=money("100"))
        assert txn.qty == 100
        assert txn.net_cash < 0

    def test_sell_direction_is_derived(self, service: AccountService) -> None:
        service.deposit(money("100000"))
        service.trade(symbol=MAOTAI, side="buy", qty=100, price=money("100"))
        txn = service.trade(
            symbol=MAOTAI, side="sell", qty=100, price=money("110"), trade_date=dt.date(2026, 7, 25)
        )
        assert txn.qty == -100
        assert txn.net_cash > 0

    @pytest.mark.parametrize("side", ["long", "SHORT", ""])
    def test_bad_side_rejected(self, service: AccountService, side: str) -> None:
        with pytest.raises(LedgerError, match="buy 或 sell"):
            service.trade(symbol=MAOTAI, side=side, qty=100, price=money("100"))

    def test_non_positive_qty_rejected(self, service: AccountService) -> None:
        with pytest.raises(LedgerError, match="必须为正"):
            service.trade(symbol=MAOTAI, side="buy", qty=-100, price=money("100"))

    def test_illegal_transaction_never_reaches_disk(self, service: AccountService) -> None:
        # **这是最关键的一条**：非法流水一旦落盘，之后每次重放都会抛错，
        # 账本就彻底打不开了。必须先在内存里验一遍
        service.deposit(money("100000"))
        service.trade(symbol=MAOTAI, side="buy", qty=100, price=money("100"))
        before = service.store.count()

        with pytest.raises(LedgerError):
            service.trade(
                symbol=MAOTAI,
                side="sell",
                qty=999,  # 超过持仓
                price=money("110"),
                trade_date=dt.date(2026, 7, 25),
            )

        assert service.store.count() == before
        assert service.ledger() is not None  # 账本仍能正常打开

    def test_replay_is_reproducible(self, service: AccountService) -> None:
        # 红线 R8：持仓完全由流水重放得出，随时可重建
        service.deposit(money("100000"))
        service.trade(symbol=MAOTAI, side="buy", qty=100, price=money("100"))
        service.trade(symbol=PINGAN, side="buy", qty=200, price=money("50"))

        first = service.state()
        second = service.state()
        assert first == second

    def test_as_of_reconstructs_history(self, service: AccountService) -> None:
        service.deposit(money("100000"), trade_date=dt.date(2026, 7, 1))
        service.trade(
            symbol=MAOTAI, side="buy", qty=100, price=money("100"), trade_date=dt.date(2026, 7, 2)
        )
        service.trade(
            symbol=PINGAN, side="buy", qty=200, price=money("50"), trade_date=dt.date(2026, 7, 20)
        )

        early = service.positions(as_of=dt.date(2026, 7, 10))
        assert set(early) == {MAOTAI}
        assert set(service.positions()) == {MAOTAI, PINGAN}

    def test_summary_flags_unpriced_positions(self, service: AccountService) -> None:
        # 缺现价的标的按 0 计入市值。不显式列出的话，
        # 总资产会莫名少一截而用户会以为是自己算错了
        service.deposit(money("100000"))
        service.trade(symbol=MAOTAI, side="buy", qty=100, price=money("100"))

        summary = service.summary({})

        assert summary.unpriced_symbols == ("600519.SH",)
        assert "缺现价" in summary.message

    def test_summary_computes_unrealized_pnl(self, service: AccountService) -> None:
        service.deposit(money("100000"))
        service.trade(symbol=MAOTAI, side="buy", qty=100, price=money("100"))

        summary = service.summary({MAOTAI: money("120")})

        assert summary.market_value == money("12000")
        assert summary.unrealized_pnl == money("2000")
        assert summary.unpriced_symbols == ()

    def test_deposit_rejects_non_positive(self, service: AccountService) -> None:
        with pytest.raises(LedgerError, match="必须为正"):
            service.deposit(money("-100"))

    def test_withdraw_is_negative_cash(self, service: AccountService) -> None:
        service.deposit(money("100000"))
        txn = service.withdraw(money("30000"))
        assert txn.net_cash == money("-30000")

    def test_transactions_are_newest_first(self, service: AccountService) -> None:
        service.deposit(money("100000"), trade_date=dt.date(2026, 7, 1))
        service.trade(
            symbol=MAOTAI, side="buy", qty=100, price=money("100"), trade_date=dt.date(2026, 7, 20)
        )
        records = service.transactions()
        assert records[0].trade_date > records[-1].trade_date

    def test_import_is_all_or_nothing(self, service: AccountService) -> None:
        # 导入一半的对账单比完全没导入更难收拾
        service.deposit(money("1000000"))
        before = service.store.count()

        with pytest.raises(LedgerError):
            service.import_transactions(
                [
                    {
                        "symbol": "600519.SH",
                        "side": "buy",
                        "qty": 100,
                        "price": "100",
                        "trade_date": "2026-07-02",
                    },
                    {
                        "symbol": "600519.SH",
                        "side": "sell",
                        "qty": 9999,
                        "price": "110",
                        "trade_date": "2026-07-03",
                    },
                ]
            )

        assert service.store.count() == before

    def test_import_success(self, service: AccountService) -> None:
        service.deposit(money("1000000"))
        written = service.import_transactions(
            [
                {
                    "symbol": "600519.SH",
                    "side": "buy",
                    "qty": 100,
                    "price": "100",
                    "trade_date": "2026-07-02",
                },
                {
                    "symbol": "601318.SH",
                    "side": "buy",
                    "qty": 200,
                    "price": "50",
                    "trade_date": "2026-07-03",
                },
            ]
        )
        assert written == 2
        assert set(service.positions()) == {MAOTAI, PINGAN}

    def test_tax_countdown_needs_lot_level_records(self, service: AccountService) -> None:
        # 只存平均成本算不出这个：平均成本里没有建仓日。
        # 对高股息标的，差几天卖掉红利税从免征跳到 10%，可能值几千元
        service.deposit(money("100000"))
        service.trade(
            symbol=MAOTAI, side="buy", qty=100, price=money("100"), trade_date=dt.date(2026, 7, 1)
        )

        countdown = service.tax_countdown(as_of=dt.date(2026, 7, 24))

        assert MAOTAI in countdown
        assert 300 < countdown[MAOTAI] <= 365


class TestAdvisorSeesTheLedger:
    """建议服务必须真的看到账本。

    这是整个改动的**目的**。补了流水存储却没接上建议服务，
    等于只是多了一个记账本，建议还是按空账户出。
    """

    def test_advisor_picks_up_recorded_positions(self, tmp_path: Path) -> None:
        from quantstock.services.advisor_service import AdvisorService  # noqa: PLC0415

        config = RootConfig()
        config.app.var_dir = str(tmp_path / "var")
        config.intel.enabled = False
        settings = Settings(
            config=config, secrets=Secrets(_env_file=None), config_dir=tmp_path / "cfg"
        )

        account = AccountService(settings)
        account.deposit(money("100000"))
        account.trade(symbol=MAOTAI, side="buy", qty=100, price=money("100"))

        advisor = AdvisorService(settings)

        assert advisor._ledger is not None
        assert MAOTAI in advisor._ledger.positions()

    def test_cold_start_still_works(self, tmp_path: Path) -> None:
        # 没有流水时必须仍是冷启动，而不是崩掉——新用户第一次打开就该能用
        from quantstock.services.advisor_service import AdvisorService  # noqa: PLC0415

        config = RootConfig()
        config.app.var_dir = str(tmp_path / "var")
        config.intel.enabled = False
        settings = Settings(
            config=config, secrets=Secrets(_env_file=None), config_dir=tmp_path / "cfg"
        )

        assert AdvisorService(settings)._ledger is None
