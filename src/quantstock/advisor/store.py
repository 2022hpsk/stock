"""交易计划的落盘与读回（红线 R6：可追溯可复现）。

计划生成（T 日收盘后）与执行（T+1 开盘）通常是两个进程、两个时间点，
计划必须能落盘。同时这份文件就是**审计快照**——它带着数据指纹、
策略版本、参数哈希和完整的四支柱解释，任何一天的建议都能凭它复盘：
"当时为什么建议买"和"当时看到的证据是什么"都在里面。

**只追加不修改**：同一 ``plan_id`` 重复写入会被拒绝，确认信息通过
``mark_confirmed`` 单独落一份 ``.confirm.json``，原始计划文件永不改动
（与红线 R8 的账本同一思路）。
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from pathlib import Path

from quantstock.advisor.types import TradePlan
from quantstock.infra.clock import now
from quantstock.infra.errors import QuantStockError
from quantstock.infra.logging import get_logger
from quantstock.infra.serde import from_jsonable, to_jsonable
from quantstock.infra.types import PlanId, TradeDate

__all__ = ["PlanNotFoundError", "PlanStore"]

_log = get_logger(__name__)

SCHEMA_VERSION = 1


class PlanNotFoundError(QuantStockError):
    """计划文件不存在。"""


class PlanStore:
    """计划文件仓库。

    目录结构::

        var/plans/2026-07-24/plan-<plan_id>.json
        var/plans/2026-07-24/plan-<plan_id>.confirm.json
    """

    def __init__(self, root: Path) -> None:
        """初始化。

        Args:
            root: 计划根目录，通常是 ``var/plans``。
        """
        self._root = Path(root)

    def _dir(self, trade_date: TradeDate) -> Path:
        """某交易日的目录。

        Args:
            trade_date: 交易日。

        Returns:
            目录路径。
        """
        return self._root / trade_date.isoformat()

    def path_of(self, plan: TradePlan) -> Path:
        """计划文件路径。

        Args:
            plan: 交易计划。

        Returns:
            文件路径。
        """
        return self._dir(plan.trade_date) / f"plan-{plan.plan_id}.json"

    def save(self, plan: TradePlan) -> Path:
        """落盘计划。

        Args:
            plan: 交易计划。

        Returns:
            写入的文件路径。

        Raises:
            QuantStockError: 该 plan_id 已存在。
        """
        path = self.path_of(plan)
        if path.exists():
            msg = "计划已存在，不允许覆盖（审计快照只追加不修改）"
            raise QuantStockError(msg, plan_id=plan.plan_id, path=str(path))

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": now().isoformat(),
            "plan": to_jsonable(plan),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _log.info("plan_saved", plan_id=plan.plan_id, path=str(path), intents=len(plan.intents))
        return path

    def load(self, trade_date: TradeDate, plan_id: PlanId) -> TradePlan:
        """读回计划。

        Args:
            trade_date: 交易日。
            plan_id: 计划标识。

        Returns:
            交易计划。

        Raises:
            PlanNotFoundError: 文件不存在。
            QuantStockError: schema 版本不匹配。
        """
        path = self._dir(trade_date) / f"plan-{plan_id}.json"
        if not path.exists():
            msg = "计划文件不存在"
            raise PlanNotFoundError(msg, plan_id=plan_id, path=str(path))

        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            msg = "计划文件 schema 版本不匹配，拒绝按旧格式执行"
            raise QuantStockError(msg, expected=SCHEMA_VERSION, actual=version, path=str(path))

        plan = from_jsonable(TradePlan, payload["plan"])
        confirm = path.with_suffix(".confirm.json")
        if confirm.exists():
            info = json.loads(confirm.read_text(encoding="utf-8"))
            plan = replace(
                plan,
                confirmed_by=str(info["confirmed_by"]),
                confirmed_at=dt.datetime.fromisoformat(str(info["confirmed_at"])),
            )
        return plan

    def latest(self, trade_date: TradeDate) -> TradePlan | None:
        """取某交易日最近保存的计划。

        Args:
            trade_date: 交易日。

        Returns:
            计划；当日无计划时 None。
        """
        folder = self._dir(trade_date)
        if not folder.is_dir():
            return None
        files = sorted(
            (p for p in folder.glob("plan-*.json") if not p.name.endswith(".confirm.json")),
            key=lambda p: p.stat().st_mtime,
        )
        if not files:
            return None
        plan_id = PlanId(files[-1].stem.removeprefix("plan-"))
        return self.load(trade_date, plan_id)

    def list_dates(self) -> list[TradeDate]:
        """列出有计划的交易日，供审计页翻阅。

        Returns:
            升序的交易日列表。
        """
        if not self._root.is_dir():
            return []
        dates: list[TradeDate] = []
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            try:
                dates.append(dt.date.fromisoformat(child.name))
            except ValueError:
                continue  # 目录名不是日期，不是计划目录
        return sorted(dates)

    def mark_confirmed(self, plan: TradePlan, *, confirmed_by: str) -> TradePlan:
        """记录人工确认（红线 R5）。

        确认信息写在**独立文件**里，原始计划文件不被修改。

        Args:
            plan: 交易计划。
            confirmed_by: 确认人。

        Returns:
            带确认信息的计划。

        Raises:
            QuantStockError: 确认人为空。
        """
        if not confirmed_by.strip():
            msg = "必须记录确认人（红线 R5）"
            raise QuantStockError(msg, plan_id=plan.plan_id)

        moment = now()
        path = self.path_of(plan).with_suffix(".confirm.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"confirmed_by": confirmed_by, "confirmed_at": moment.isoformat()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _log.info("plan_confirmed", plan_id=plan.plan_id, confirmed_by=confirmed_by)
        return replace(plan, confirmed_by=confirmed_by, confirmed_at=moment)
