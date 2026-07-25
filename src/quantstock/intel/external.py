"""外置信息导入（★ 用户明确要求的能力）。

规范见 docs/07-信息情报模块.md 第五节。

四种入口统一落到 ``IntelItem``：

1. **收件箱目录** ``var/intel/inbox/``——把文件丢进去即可，零门槛；
2. **CLI 直接录入** ``quantstock intel note``；
3. 本地 HTTP 接收端（在 ``web`` 层实现，复用本模块的解析器）；
4. 自定义源插件（见 ``intel/protocols.py``）。

**处理过的文件一定要移走**，移到 ``_processed/<date>/``。留在原地会被反复解析，
虽然 ``item_id`` 幂等挡得住重复入库，但每次采集都重扫全量文件迟早会拖垮任务。
解析失败的移到 ``_failed/`` 并**在旁边写一份 ``.error.txt``**——
文件静静消失是最让人困惑的失败方式。
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from quantstock.infra.clock import CST, now
from quantstock.infra.errors import IntelError
from quantstock.infra.logging import get_logger
from quantstock.infra.types import Symbol, parse_symbol
from quantstock.intel.dedup import content_hash
from quantstock.intel.types import EventType, IntelDomain, IntelItem, SourceTier

_EnumT = TypeVar("_EnumT", IntelDomain, EventType)

__all__ = [
    "INBOX_NAMESPACE",
    "ImportReport",
    "InboxScanner",
    "build_item",
    "items_from_rows",
    "parse_payload",
]

_log = get_logger(__name__)

INBOX_NAMESPACE = uuid.UUID("6f1a5c2e-9a3d-5f47-9c1e-7b2d4a8e6f30")
"""``item_id`` 的 uuid5 命名空间。固定值，保证跨进程、跨机器派生出同一个 id。"""

SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".json", ".csv"})
_FRONT_MATTER = "---"
_DATE_PREFIX_PARTS = 4
"""``2026-07-25-标题`` 切成 4 段后，末段才是标题。"""


def make_item_id(source: str, url: str, digest: str) -> str:
    """派生幂等的条目 id。

    Args:
        source: 来源标识。
        url: 原文链接，可为空。
        digest: 内容指纹。

    Returns:
        uuid5 的十六进制串。
    """
    return uuid.uuid5(INBOX_NAMESPACE, f"{source}\x00{url}\x00{digest}").hex


def _coerce_datetime(value: Any, fallback: dt.datetime) -> dt.datetime:  # noqa: ANN401 - 来自用户输入
    """把用户给的时间转成 tz-aware。

    未带时区的一律按 Asia/Shanghai 解释——用户手写"2026-07-25 09:00"
    时想表达的是本地时间，按 UTC 解释会把它推后 8 小时（红线 R3）。

    Args:
        value: 用户输入。
        fallback: 缺省值。

    Returns:
        tz-aware 时间。
    """
    if value is None or value == "":
        return fallback
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=CST)
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time(), tzinfo=CST)
    try:
        parsed = dt.datetime.fromisoformat(str(value))
    except ValueError:
        return fallback
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=CST)


def _coerce_symbols(value: Any) -> tuple[Symbol, ...]:  # noqa: ANN401 - 来自用户输入
    """归一化标的列表。

    Args:
        value: 字符串、列表或 None。

    Returns:
        标准 Symbol 元组。无法识别的条目被丢弃而非报错——
        一条格式不对的代码不该让整份导入失败。
    """
    if not value:
        return ()
    raw = value.split(",") if isinstance(value, str) else list(value)
    out: list[Symbol] = []
    for entry in raw:
        text = str(entry).strip()
        if not text:
            continue
        try:
            out.append(parse_symbol(text))
        except ValueError:
            _log.warning("external_symbol_unparseable", value=text)
    return tuple(dict.fromkeys(out))


def _coerce_enum(enum_cls: type[_EnumT], value: Any, default: _EnumT | None) -> _EnumT | None:  # noqa: ANN401 - 来自用户输入
    """宽松地解析枚举值。

    大小写不敏感——用户在 front-matter 里写 ``POLICY`` 和 ``policy``
    都应该工作。

    Args:
        enum_cls: 枚举类。
        value: 用户输入。
        default: 缺省值。

    Returns:
        枚举成员；无法识别时返回缺省值。
    """
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    for member in enum_cls:
        if member.value == text or member.name.lower() == text:
            return member
    _log.warning("external_enum_unknown", enum=enum_cls.__name__, value=str(value))
    return default


def build_item(
    payload: dict[str, Any],
    *,
    source: str = "external:inbox",
    fetched_at: dt.datetime | None = None,
) -> IntelItem:
    """把一份宽松的字典转成 ``IntelItem``。

    宽松模式：缺省字段自动补全。这是外置导入的核心体验——
    用户丢进来一句话也应该能入库，而不是先学一套 schema。

    Args:
        payload: 用户提供的字段。
        source: 来源标识。
        fetched_at: 抓取时刻。

    Returns:
        标准化的情报条目。

    Raises:
        IntelError: 标题与正文都为空。
    """
    moment = fetched_at or now()
    title = str(payload.get("title") or "").strip()
    body = str(payload.get("body") or payload.get("content") or "").strip()
    if not title and not body:
        msg = "标题与正文不能同时为空"
        raise IntelError(msg, source=source)
    if not title:
        # 只给了正文时用首行当标题，比留空好——日报里总得有个能显示的东西
        title = body.splitlines()[0][:80]

    digest = content_hash(title, body)
    url = str(payload.get("url") or "").strip()
    tier = payload.get("source_tier")
    importance = payload.get("importance")

    return IntelItem(
        item_id=str(payload.get("item_id") or make_item_id(source, url, digest)),
        source=str(payload.get("source") or source),
        source_tier=_coerce_tier(tier),
        domain=_coerce_enum(IntelDomain, payload.get("domain"), IntelDomain.COMPANY)
        or IntelDomain.COMPANY,
        domain_declared=bool(str(payload.get("domain") or "").strip()),
        publish_at=_coerce_datetime(payload.get("publish_at"), moment),
        fetched_at=moment,
        title=title,
        content_hash=digest,
        body=body,
        url=url,
        symbols=_coerce_symbols(payload.get("symbols")),
        industries=tuple(str(x) for x in payload.get("industries", ()) if str(x).strip()),
        themes=tuple(str(x) for x in payload.get("themes", ()) if str(x).strip()),
        event_type=_coerce_enum(EventType, payload.get("event_type") or payload.get("event"), None),
        importance=max(0, min(100, int(importance))) if importance not in (None, "") else 0,
        sentiment=float(payload.get("sentiment") or 0.0),
        raw={"origin": source, **{k: str(v) for k, v in payload.items() if k == "path"}},
    )


def _coerce_tier(value: Any) -> SourceTier:  # noqa: ANN401 - 来自用户输入
    """解析来源层级。

    Args:
        value: 用户输入。

    Returns:
        层级；无法识别时按 USER——人工导入的默认身份，
        它带着 importance 上限，是最保守的一侧。
    """
    if value is None or value == "":
        return SourceTier.USER
    if isinstance(value, SourceTier):
        return value
    text = str(value).strip().upper()
    for member in SourceTier:
        if member.name == text or str(int(member)) == text:
            return member
    return SourceTier.USER


def parse_payload(path: Path, text: str) -> list[dict[str, Any]]:
    """按扩展名解析文件内容。

    Args:
        path: 文件路径，用于取扩展名与缺省标题。
        text: 文件内容。

    Returns:
        零到多份字段字典。

    Raises:
        IntelError: 格式非法或不支持的扩展名。
    """
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return [_parse_markdown(path, text)]
    if suffix == ".json":
        return _parse_json(text)
    if suffix == ".csv":
        return _parse_csv(text)
    msg = f"不支持的文件类型 {suffix}"
    raise IntelError(msg, path=str(path))


def _parse_markdown(path: Path, text: str) -> dict[str, Any]:
    """解析 Markdown / 纯文本，支持 YAML front-matter。

    Args:
        path: 文件路径。
        text: 内容。

    Returns:
        字段字典。

    Raises:
        IntelError: front-matter 不是合法 YAML 映射。
    """
    meta: dict[str, Any] = {}
    body = text
    if text.lstrip().startswith(_FRONT_MATTER):
        stripped = text.lstrip()
        end = stripped.find(f"\n{_FRONT_MATTER}", len(_FRONT_MATTER))
        if end != -1:
            head = stripped[len(_FRONT_MATTER) : end]
            body = stripped[end + len(_FRONT_MATTER) + 1 :].lstrip("\n")
            try:
                loaded = yaml.safe_load(head) or {}
            except yaml.YAMLError as exc:
                msg = "front-matter 不是合法 YAML"
                raise IntelError(msg, path=str(path), error=str(exc)) from exc
            if not isinstance(loaded, dict):
                msg = "front-matter 必须是键值映射"
                raise IntelError(msg, path=str(path))
            meta = loaded

    meta.setdefault("title", _title_from_filename(path))
    meta.setdefault("body", body.strip())
    return meta


def _title_from_filename(path: Path) -> str:
    """从文件名推标题。

    ``2026-07-25-央行降准.md`` → ``央行降准``。前缀日期是给人排序看的，
    不该出现在日报标题里。

    Args:
        path: 文件路径。

    Returns:
        标题。
    """
    stem = path.stem
    parts = stem.split("-", 3)
    if len(parts) == _DATE_PREFIX_PARTS and parts[0].isdigit():
        return parts[3]
    return stem


def _parse_json(text: str) -> list[dict[str, Any]]:
    """解析 JSON，单条或数组皆可。

    Args:
        text: 内容。

    Returns:
        字段字典列表。

    Raises:
        IntelError: 非法 JSON 或结构不符。
    """
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = "非法 JSON"
        raise IntelError(msg, error=str(exc)) from exc
    rows = loaded if isinstance(loaded, list) else [loaded]
    if not all(isinstance(r, dict) for r in rows):
        msg = "JSON 必须是对象或对象数组"
        raise IntelError(msg)
    return list(rows)


def _parse_csv(text: str) -> list[dict[str, Any]]:
    """解析 CSV。

    Args:
        text: 内容。

    Returns:
        字段字典列表。
    """
    reader = csv.DictReader(io.StringIO(text))
    return [{k: v for k, v in row.items() if k} for row in reader]


@dataclass(frozen=True, slots=True)
class ImportReport:
    """一次收件箱扫描的结果。"""

    items: tuple[IntelItem, ...]
    processed: tuple[Path, ...]
    failed: tuple[tuple[Path, str], ...]

    @property
    def summary(self) -> str:
        """人类可读摘要。"""
        base = f"导入 {len(self.items)} 条，处理 {len(self.processed)} 个文件"
        return base + (f"，失败 {len(self.failed)} 个" if self.failed else "")


class InboxScanner:
    """收件箱目录扫描器。"""

    def __init__(self, inbox: Path) -> None:
        """初始化。

        Args:
            inbox: 收件箱目录，通常是 ``var/intel/inbox``。
        """
        self._inbox = Path(inbox)
        self._processed = self._inbox / "_processed"
        self._failed = self._inbox / "_failed"

    @property
    def inbox_dir(self) -> Path:
        """收件箱目录。"""
        return self._inbox

    def ensure_dirs(self) -> None:
        """建好目录结构，并放一份说明文件。"""
        self._inbox.mkdir(parents=True, exist_ok=True)
        readme = self._inbox / "README.md"
        if not readme.exists():
            readme.write_text(_INBOX_README, encoding="utf-8")

    def pending(self) -> Iterator[Path]:
        """列出待处理文件。

        Yields:
            文件路径，按名称排序以保证可复现。
        """
        if not self._inbox.is_dir():
            return
        for path in sorted(self._inbox.iterdir()):
            if path.is_dir() or path.name.startswith((".", "_")):
                continue
            if path.name == "README.md":
                continue
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                yield path

    def scan(self, *, move: bool = True) -> ImportReport:
        """扫描并导入。

        Args:
            move: 处理后是否移走文件。设 False 便于测试与预览。

        Returns:
            导入报告。
        """
        items: list[IntelItem] = []
        processed: list[Path] = []
        failed: list[tuple[Path, str]] = []
        stamp = now()

        for path in self.pending():
            try:
                text = path.read_text(encoding="utf-8")
                payloads = parse_payload(path, text)
                parsed = [build_item({**p, "path": str(path)}, fetched_at=stamp) for p in payloads]
            except (OSError, IntelError, ValueError) as exc:
                failed.append((path, str(exc)))
                if move:
                    self._move_failed(path, str(exc))
                _log.warning("intel_inbox_failed", path=str(path), error=str(exc))
                continue

            items.extend(parsed)
            processed.append(path)
            if move:
                self._move_processed(path, stamp)

        if items or failed:
            _log.info("intel_inbox_scanned", items=len(items), failed=len(failed))
        return ImportReport(items=tuple(items), processed=tuple(processed), failed=tuple(failed))

    def _move_processed(self, path: Path, stamp: dt.datetime) -> Path:
        """把处理完的文件归档。

        Args:
            path: 源文件。
            stamp: 处理时刻，用于分目录。

        Returns:
            归档后的路径。
        """
        target_dir = self._processed / stamp.date().isoformat()
        target_dir.mkdir(parents=True, exist_ok=True)
        return _move_without_clobber(path, target_dir / path.name)

    def _move_failed(self, path: Path, error: str) -> Path:
        """把失败的文件移走并写下原因。

        Args:
            path: 源文件。
            error: 错误说明。

        Returns:
            移动后的路径。
        """
        self._failed.mkdir(parents=True, exist_ok=True)
        target = _move_without_clobber(path, self._failed / path.name)
        target.with_suffix(target.suffix + ".error.txt").write_text(error, encoding="utf-8")
        return target


def _move_without_clobber(source: Path, target: Path) -> Path:
    """移动文件，同名时加后缀而非覆盖。

    Args:
        source: 源路径。
        target: 目标路径。

    Returns:
        实际写入的路径。
    """
    final = target
    counter = 1
    while final.exists():
        final = target.with_name(f"{target.stem}-{counter}{target.suffix}")
        counter += 1
    source.rename(final)
    return final


def items_from_rows(rows: Sequence[dict[str, Any]], *, source: str) -> list[IntelItem]:
    """把一批字典转成条目，跳过无法解析的行。

    供 CLI ``intel import`` 与 HTTP 接收端复用。

    Args:
        rows: 字段字典。
        source: 来源标识。

    Returns:
        情报条目列表。
    """
    out: list[IntelItem] = []
    for row in rows:
        try:
            out.append(build_item(row, source=source))
        except (IntelError, ValueError) as exc:
            _log.warning("external_row_skipped", error=str(exc))
    return out


_INBOX_README = """# 情报收件箱

把文件丢进这个目录，下次采集任务会自动吸收，处理后移到 `_processed/<日期>/`。
解析失败的移到 `_failed/`，旁边会有一份 `.error.txt` 说明原因。

支持 `.md` / `.txt` / `.json` / `.csv`。

## 最简用法

新建 `随手记.md`，写一句话就行：

```
中芯国际发布新产能规划，月产能提升 20%
```

## 带元数据（Markdown front-matter）

```markdown
---
domain: POLICY
publish_at: 2026-07-25T09:00:00+08:00
symbols: [601398.SH, 601939.SH]
importance: 85
url: https://example.com/xxx
title: 央行宣布降准 0.5 个百分点
---
正文内容……
```

字段全部可选。`publish_at` 不写时用文件处理时刻；不带时区时按 Asia/Shanghai 解释。

## 注意

人工导入的条目 `source_tier` 为 `USER`，`importance` 上限 90——
这是刻意的：防止单条手工输入压过全部量化信号。
"""
