"""数据类 ↔ JSON 的通用编解码。

**为什么要自己写而不是用 pydantic**：``advisor`` / ``execution`` 层的契约刻意用
frozen dataclass 而非 pydantic 模型——它们是内部值对象，不需要运行期校验开销，
也不应该被"外部输入"的语义污染。但红线 R6 要求每条建议可追溯可复现，
计划必须能落盘再读回，于是需要一层薄的编解码。

编码规则刻意保守：

- ``Decimal`` 编成**字符串**而非数字。JSON 数字会被解析成 float，
  一旦经过 float 就再也回不到精确值了（红线 R1）。
- ``datetime`` 编成带时区的 ISO 串；解码时**断言必须带时区**（红线 R3）。
  丢了时区的时间戳在跨时区回放时会静默算错。
- 未知字段解码时报错而不是忽略——schema 变了要立刻知道。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import types
import typing
from decimal import Decimal
from enum import Enum
from typing import Any, TypeVar, get_args, get_origin

from quantstock.infra.errors import QuantStockError

__all__ = ["SerdeError", "from_jsonable", "to_jsonable"]

T = TypeVar("T")


class SerdeError(QuantStockError):
    """编解码失败。"""


def to_jsonable(value: object) -> Any:  # noqa: ANN401 - 递归编码，返回类型必然是 Any
    """把值编码成可直接 ``json.dumps`` 的结构。

    Args:
        value: 待编码的值。

    Returns:
        由 dict/list/str/int/float/bool/None 组成的结构。

    Raises:
        SerdeError: 遇到不支持的类型。
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in value]
    return _encode_scalar(value)


def _encode_scalar(value: object) -> Any:  # noqa: ANN401 - 多种标量映射到 JSON 基本类型
    """编码标量。

    Args:
        value: 待编码的标量。

    Returns:
        JSON 基本类型。

    Raises:
        SerdeError: 类型不支持，或时间戳缺时区。
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        # 走字符串，绝不让金额经过 float（红线 R1）
        return str(value)
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            msg = "禁止序列化 naive datetime（红线 R3）"
            raise SerdeError(msg, value=value.isoformat())
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    msg = "不支持的编码类型"
    raise SerdeError(msg, type=type(value).__name__)


def from_jsonable(target: type[T], data: Any) -> T:  # noqa: ANN401 - 输入来自 json.loads
    """按目标类型解码 JSON 结构。

    Args:
        target: 目标类型，通常是 dataclass。
        data: ``json.loads`` 的结果。

    Returns:
        目标类型的实例。

    Raises:
        SerdeError: 结构与目标类型不匹配。
    """
    return typing.cast("T", _decode(target, data))


# ------------------------------------------------------------------ 内部
def _unwrap_newtype(target: Any) -> Any:  # noqa: ANN401 - 类型对象本身
    """展开 ``NewType``。

    Args:
        target: 可能是 NewType 的类型对象。

    Returns:
        底层类型。
    """
    while hasattr(target, "__supertype__"):
        target = target.__supertype__
    return target


def _decode(target: Any, data: Any) -> Any:  # noqa: ANN401, C901, PLR0911, PLR0912 - 类型分发天然分支多
    """按类型递归解码。

    Args:
        target: 目标类型。
        data: 待解码数据。

    Returns:
        解码结果。

    Raises:
        SerdeError: 结构不匹配或类型不支持。
    """
    target = _unwrap_newtype(target)
    origin = get_origin(target)

    if origin in (types.UnionType, typing.Union):
        args = [a for a in get_args(target) if a is not type(None)]
        if data is None:
            return None
        if len(args) != 1:
            msg = "只支持 X | None 形式的联合类型"
            raise SerdeError(msg, target=str(target))
        return _decode(args[0], data)

    if origin in (list, tuple, set, frozenset):
        item_type = next((a for a in get_args(target) if a is not Ellipsis), Any)
        if not isinstance(data, list):
            msg = "期望数组"
            raise SerdeError(msg, target=str(target), got=type(data).__name__)
        items = [_decode(item_type, x) for x in data]
        return origin(items)

    if origin is dict:
        key_type, value_type = get_args(target) or (str, Any)
        if not isinstance(data, dict):
            msg = "期望对象"
            raise SerdeError(msg, target=str(target), got=type(data).__name__)
        return {_decode(key_type, k): _decode(value_type, v) for k, v in data.items()}

    if target is Any:
        return data
    if isinstance(target, type):
        if issubclass(target, Enum):
            return target(data)
        if issubclass(target, bool):
            return bool(data)
        if issubclass(target, Decimal):
            return Decimal(str(data))
        if issubclass(target, dt.datetime):
            return _decode_datetime(data)
        if issubclass(target, dt.date):
            return dt.date.fromisoformat(str(data))
        if issubclass(target, str | int | float):
            return target(data)
        if dataclasses.is_dataclass(target):
            return _decode_dataclass(target, data)

    msg = "不支持的解码类型"
    raise SerdeError(msg, target=str(target))


def _decode_datetime(data: Any) -> dt.datetime:  # noqa: ANN401 - 来自 json
    """解码时间戳并强制带时区。

    Args:
        data: ISO 字符串。

    Returns:
        tz-aware 时间。

    Raises:
        SerdeError: 缺少时区信息。
    """
    moment = dt.datetime.fromisoformat(str(data))
    if moment.tzinfo is None:
        msg = "时间戳缺少时区信息（红线 R3）"
        raise SerdeError(msg, value=str(data))
    return moment


def _decode_dataclass(target: type[Any], data: Any) -> Any:  # noqa: ANN401 - 来自 json
    """解码 dataclass。

    Args:
        target: 目标 dataclass 类型。
        data: JSON 对象。

    Returns:
        dataclass 实例。

    Raises:
        SerdeError: 结构不匹配或含未知字段。
    """
    if not isinstance(data, dict):
        msg = "期望对象"
        raise SerdeError(msg, target=target.__name__, got=type(data).__name__)

    hints = typing.get_type_hints(target)
    names = {f.name for f in dataclasses.fields(target)}
    if unknown := sorted(set(data) - names):
        # 静默忽略未知字段会让 schema 漂移到发现时已经错了很多天
        msg = "存在未知字段，schema 可能已变更"
        raise SerdeError(msg, target=target.__name__, unknown=unknown)

    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(target):
        if field.name not in data:
            continue
        kwargs[field.name] = _decode(hints[field.name], data[field.name])
    return target(**kwargs)
