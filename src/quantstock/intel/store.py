"""情报存储。

规范见 docs/07-信息情报模块.md 第九节。

```
var/lake/intel/items/date=2026-07-25/part.json
var/lake/intel/digests/2026-07-25-post.json
var/lake/intel/blacklist/state.json
```

按 ``publish_at`` 的自然日分区。**分区键必须用发布日而非抓取日**——
回测按"当时看得到什么"取数，抓取日分区会让今天回补的历史情报全落到今天的分区里，
`as_of` 过滤就形同虚设了（红线 I-R5）。

主键 ``item_id``，写入幂等：同一条重复入库只保留一份，且合并 ``duplicates``。
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from quantstock.infra.logging import get_logger
from quantstock.infra.serde import from_jsonable, to_jsonable
from quantstock.infra.types import TradeDate
from quantstock.intel.types import IntelDigest, IntelItem

__all__ = ["IntelStore"]

_log = get_logger(__name__)

SCHEMA_VERSION = 1


class IntelStore:
    """情报数据湖。

    落 JSON 而非 Parquet：情报量级是每天几百到几千条，JSON 的可读性
    在排查"这条消息当时到底入没入库"时价值远大于压缩率。
    条目里含嵌套的 ``raw`` 字典，Parquet 的列式结构对它并不友好。
    """

    def __init__(self, root: Path) -> None:
        """初始化。

        Args:
            root: 数据湖根目录，通常是 ``var/lake/intel``。
        """
        self._root = Path(root)
        self._items = self._root / "items"
        self._digests = self._root / "digests"

    @property
    def blacklist_path(self) -> Path:
        """黑名单状态文件路径。"""
        return self._root / "blacklist" / "state.json"

    def partition(self, trade_date: TradeDate) -> Path:
        """某日的分区文件。

        Args:
            trade_date: 发布日。

        Returns:
            文件路径。
        """
        return self._items / f"date={trade_date.isoformat()}" / "part.json"

    def write(self, items: Iterable[IntelItem]) -> int:
        """写入情报，按发布日分区，幂等。

        Args:
            items: 待写入条目。

        Returns:
            实际新增的条数（已存在的不计）。
        """
        by_date: dict[TradeDate, list[IntelItem]] = {}
        for item in items:
            by_date.setdefault(item.trade_date, []).append(item)

        added = 0
        for trade_date, batch in by_date.items():
            added += self._write_partition(trade_date, batch)
        return added

    def _write_partition(self, trade_date: TradeDate, batch: Sequence[IntelItem]) -> int:
        """写单个分区。

        Args:
            trade_date: 分区日。
            batch: 该日条目。

        Returns:
            新增条数。
        """
        path = self.partition(trade_date)
        existing = {i.item_id: i for i in self._read_partition(path)}
        before = len(existing)

        for item in batch:
            prior = existing.get(item.item_id)
            if prior is None:
                existing[item.item_id] = item
                continue
            # 重复入库：合并 duplicates 并取更高的 importance。
            # 多源印证会随时间陆续到达，后来的印证不该被先到的版本覆盖掉。
            existing[item.item_id] = replace(
                item,
                duplicates=tuple(dict.fromkeys([*prior.duplicates, *item.duplicates])),
                importance=max(prior.importance, item.importance),
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "date": trade_date.isoformat(),
            "items": [to_jsonable(i) for i in sorted(existing.values(), key=_sort_key)],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        added = len(existing) - before
        _log.info(
            "intel_items_written", date=trade_date.isoformat(), added=added, total=len(existing)
        )
        return added

    @staticmethod
    def _read_partition(path: Path) -> list[IntelItem]:
        """读单个分区。

        Args:
            path: 分区文件。

        Returns:
            条目列表；文件不存在时为空。
        """
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [from_jsonable(IntelItem, row) for row in payload.get("items", [])]

    def read(self, trade_date: TradeDate) -> list[IntelItem]:
        """读某日情报。

        Args:
            trade_date: 发布日。

        Returns:
            条目列表。
        """
        return self._read_partition(self.partition(trade_date))

    def read_range(
        self, start: TradeDate, end: TradeDate, *, as_of: dt.datetime | None = None
    ) -> list[IntelItem]:
        """读一个日期区间的情报。

        Args:
            start: 起始日（含）。
            end: 结束日（含）。
            as_of: PIT 截断时点。**回测里必须传**，否则会读到决策时点之后
                才发布的消息（红线 I-R5）。

        Returns:
            按发布时间升序的条目。
        """
        out: list[IntelItem] = []
        cursor = start
        while cursor <= end:
            out.extend(self.read(cursor))
            cursor += dt.timedelta(days=1)
        if as_of is not None:
            out = [i for i in out if i.visible_at(as_of)]
        return sorted(out, key=_sort_key)

    def available_dates(self) -> list[TradeDate]:
        """已有情报的日期。

        Returns:
            升序日期列表。
        """
        if not self._items.is_dir():
            return []
        dates: list[TradeDate] = []
        for child in sorted(self._items.iterdir()):
            if not child.is_dir() or not child.name.startswith("date="):
                continue
            try:
                dates.append(dt.date.fromisoformat(child.name.removeprefix("date=")))
            except ValueError:
                continue  # 目录名不是日期，忽略
        return dates

    def save_digest(self, digest: IntelDigest) -> Path:
        """落盘摘要。

        Args:
            digest: 摘要。

        Returns:
            文件路径。
        """
        path = self._digests / f"{digest.trade_date.isoformat()}-{digest.session}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": SCHEMA_VERSION, "digest": to_jsonable(digest)}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_digest(self, trade_date: TradeDate, session: str) -> IntelDigest | None:
        """读回摘要。

        Args:
            trade_date: 交易日。
            session: ``pre`` / ``post``。

        Returns:
            摘要；不存在时 None。
        """
        path = self._digests / f"{trade_date.isoformat()}-{session}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return from_jsonable(IntelDigest, payload["digest"])

    def purge_before(self, cutoff: TradeDate) -> int:
        """清理保留期之外的分区。

        Args:
            cutoff: 保留起点，早于该日的分区被删除。

        Returns:
            删除的分区数。
        """
        removed = 0
        for trade_date in self.available_dates():
            if trade_date >= cutoff:
                continue
            folder = self.partition(trade_date).parent
            for child in folder.iterdir():
                child.unlink()
            folder.rmdir()
            removed += 1
        if removed:
            _log.info("intel_partitions_purged", removed=removed, cutoff=cutoff.isoformat())
        return removed


def _sort_key(item: IntelItem) -> tuple[dt.datetime, str]:
    """排序键：发布时间 + id，保证输出稳定可复现。

    Args:
        item: 情报条目。

    Returns:
        排序元组。
    """
    return (item.publish_at, item.item_id)
