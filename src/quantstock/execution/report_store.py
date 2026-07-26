"""执行报告的只追加落盘。

**没有它，复盘和审计都无从谈起**：``ExecutionReport`` 此前只存在于一次调用的
返回值里，进程一退就没了。于是"计划了 8 笔、实际执行 5 笔、跳过的 3 笔
分别是什么原因"这些问题，事后完全无法回答——而这正是半自动系统里
最值得回答的一类问题（docs/08 D3）。

与交易流水同样是**只追加**：执行发生过就是发生过，不能事后修改。
按交易日分文件，复盘时只读需要的那几天。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from quantstock.execution.types import ExecutionReport
from quantstock.infra.logging import get_logger
from quantstock.infra.types import TradeDate

__all__ = ["ExecutionReportStore", "report_to_json"]

_log = get_logger(__name__)


def report_to_json(report: ExecutionReport) -> dict[str, Any]:
    """执行报告转 JSON。

    **跳过的订单与提交的订单同等重要**，一起落盘：复盘要按跳过原因分组
    统计人工干预的价值，只存成交记录就永远算不出来。

    Args:
        report: 执行报告。

    Returns:
        可落盘的字典。金额一律字符串（红线 R1）。
    """
    return {
        "plan_id": str(report.plan_id),
        "trade_date": report.trade_date.isoformat(),
        "executed_at": report.executed_at.isoformat(),
        "broker": report.broker,
        "aborted": report.aborted,
        "abort_reason": report.abort_reason,
        "confirmed_by": report.confirmed_by,
        "manual_checklist": list(report.manual_checklist),
        "orders": [
            {
                "order_id": o.order_id,
                "intent_id": str(o.intent_id),
                "symbol": str(o.symbol),
                "side": str(getattr(o.side, "value", o.side)),
                "qty": o.qty,
                "price": str(o.price),
                "status": str(getattr(o.status, "value", o.status)),
                "filled_qty": o.filled_qty,
                "avg_fill_price": str(o.avg_fill_price),
                "skip_reason": str(o.skip_reason.value) if o.skip_reason else None,
                "skip_note": o.skip_note,
                "message": o.message,
            }
            for o in report.orders
        ],
        "fills": [
            {
                "fill_id": f.fill_id,
                "order_id": f.order_id,
                "symbol": str(f.symbol),
                "side": str(getattr(f.side, "value", f.side)),
                "qty": f.qty,
                "price": str(f.price),
                "fee": str(f.fee),
                "filled_at": f.filled_at.isoformat(),
            }
            for f in report.fills
        ],
    }


class ExecutionReportStore:
    """按交易日分文件的执行报告存储。"""

    def __init__(self, root: Path) -> None:
        """初始化。

        Args:
            root: 根目录。
        """
        self._root = root

    @property
    def root(self) -> Path:
        """根目录。"""
        return self._root

    def _path(self, trade_date: TradeDate) -> Path:
        """某日的报告文件。

        Args:
            trade_date: 交易日。

        Returns:
            文件路径。
        """
        return self._root / f"{trade_date.isoformat()}.jsonl"

    def append(self, report: ExecutionReport) -> Path:
        """追加一份执行报告。

        Args:
            report: 报告。

        Returns:
            落盘路径。
        """
        path = self._path(report.trade_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report_to_json(report), ensure_ascii=False) + "\n")
        _log.info(
            "execution_report_saved",
            plan_id=str(report.plan_id),
            trade_date=report.trade_date.isoformat(),
            orders=len(report.orders),
        )
        return path

    def read(self, trade_date: TradeDate) -> list[dict[str, Any]]:
        """读取某日的全部执行报告。

        Args:
            trade_date: 交易日。

        Returns:
            报告字典列表。坏行跳过并记 WARNING——复盘缺一份报告只是统计少一个
            样本，不像账本少一笔那样让结果直接变错，所以这里的取舍与流水相反。
        """
        path = self._path(trade_date)
        if not path.exists():
            return []

        out: list[dict[str, Any]] = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if not text:
                continue
            try:
                out.append(json.loads(text))
            except json.JSONDecodeError:
                _log.warning("execution_report_line_broken", path=str(path), line=lineno)
        return out

    def list_dates(self) -> list[TradeDate]:
        """有执行记录的日期。

        Returns:
            日期列表，升序。
        """
        if not self._root.exists():
            return []
        dates: list[TradeDate] = []
        for path in self._root.glob("*.jsonl"):
            try:
                dates.append(dt.date.fromisoformat(path.stem))
            except ValueError:
                continue
        return sorted(dates)
