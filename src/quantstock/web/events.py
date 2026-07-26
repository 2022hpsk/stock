"""WebSocket 事件推送。

规范见 docs/09-可视化界面规格.md 第六节：``tasks`` / ``orders`` / ``risk`` /
``intel`` / ``market`` 五个频道，断线自动重连。

**关键设计：事件先落环形缓冲，再推送**。规格第九节验收 8 要求"断开 WebSocket
后重连，任务进度与订单状态能正确恢复，无事件丢失"。只做即时广播做不到这一点——
断线那几秒的事件会永久消失，用户看到的进度条会卡在中途再也不动。
所以每个事件带一个单调递增的序号，客户端重连时带上 ``since``，
把断线期间的事件补齐。

缓冲是**有界**的（``MAX_BUFFERED``）。无界缓冲在长时间运行后会把内存吃光，
而"补齐一整天的历史"本来也不是这里该干的事——那是审计页查库的职责。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from quantstock.infra.clock import now
from quantstock.infra.logging import get_logger

__all__ = ["CHANNELS", "Event", "EventHub"]

_log = get_logger(__name__)

CHANNELS = ("tasks", "orders", "risk", "intel", "market")
"""合法频道。**白名单而非任意字符串**：拼错频道名的订阅会静默地收不到任何
消息，而这种 bug 在界面上表现为"功能没反应"，极难定位。"""

MAX_BUFFERED = 500
"""每个频道保留的历史事件数。够覆盖一次断线重连，又不会无界增长。"""

QUEUE_SIZE = 100
"""单个订阅者的待发队列长度。

满了就丢**最旧**的而不是阻塞广播：一个卡住的浏览器标签页不该让整个
推送链路停摆，更不该把发布方（回测、采集任务）拖住。
"""


@dataclass(frozen=True, slots=True)
class Event:
    """一条推送事件。"""

    seq: int
    channel: str
    kind: str
    payload: dict[str, Any]
    at: str

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 化的字典。

        Returns:
            事件字典。
        """
        return {
            "seq": self.seq,
            "channel": self.channel,
            "kind": self.kind,
            "payload": self.payload,
            "at": self.at,
        }


@dataclass(eq=False)
class _Subscriber:
    """一个订阅者。

    ``eq=False`` 是必需的：dataclass 默认生成 ``__eq__``，那会把 ``__hash__``
    置成 None，实例就放不进 ``set``。而这里要的恰恰是**按身份**区分订阅者——
    两个订阅同样频道的浏览器标签页是两个独立的连接，不该被当成同一个。
    """

    channels: frozenset[str]
    queue: asyncio.Queue[Event] = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_SIZE))
    dropped: int = 0


class EventHub:
    """事件总线。

    发布方（服务层调用点）用 ``publish`` 同步投递，订阅方（WebSocket 连接）
    用 ``subscribe`` 拿到一个异步迭代器。
    """

    def __init__(self, *, max_buffered: int = MAX_BUFFERED) -> None:
        """初始化。

        Args:
            max_buffered: 每频道保留的历史事件数。
        """
        self._seq = 0
        self._history: deque[Event] = deque(maxlen=max_buffered * len(CHANNELS))
        self._subscribers: set[_Subscriber] = set()
        self._lock = asyncio.Lock()

    @property
    def last_seq(self) -> int:
        """当前最大序号。"""
        return self._seq

    @property
    def subscriber_count(self) -> int:
        """当前订阅者数量。"""
        return len(self._subscribers)

    def publish(self, channel: str, kind: str, **payload: Any) -> Event:  # noqa: ANN401
        """发布一条事件。

        同步方法，可以从任何地方调用而不必先有事件循环——服务层的调用点
        大多是同步代码，要求它们 ``await`` 会把异步病毒式地扩散到整个后端。

        Args:
            channel: 频道名，必须在 ``CHANNELS`` 内。
            kind: 事件类型，如 ``progress`` / ``done`` / ``failed``。
            payload: 事件负载。

        Returns:
            已发布的事件。

        Raises:
            ValueError: 频道名非法。
        """
        if channel not in CHANNELS:
            msg = f"未知频道 {channel!r}，合法值：{', '.join(CHANNELS)}"
            raise ValueError(msg)

        self._seq += 1
        event = Event(
            seq=self._seq,
            channel=channel,
            kind=kind,
            payload=payload,
            at=now().isoformat(),
        )
        self._history.append(event)

        for sub in self._subscribers:
            if channel not in sub.channels:
                continue
            if sub.queue.full():
                # 丢最旧的，保证新事件总能进去。卡住的订阅者不该拖住发布方
                with contextlib.suppress(asyncio.QueueEmpty):
                    sub.queue.get_nowait()
                sub.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                sub.queue.put_nowait(event)
        return event

    def replay(self, channels: frozenset[str], *, since: int) -> list[Event]:
        """取断线期间错过的事件。

        Args:
            channels: 订阅的频道。
            since: 客户端已收到的最大序号。

        Returns:
            序号大于 ``since`` 的事件，按序号升序。
        """
        return [e for e in self._history if e.seq > since and e.channel in channels]

    @contextlib.asynccontextmanager
    async def subscribe(self, channels: frozenset[str]) -> Any:  # noqa: ANN401 - 异步上下文
        """订阅频道。

        Args:
            channels: 要订阅的频道集合。

        Yields:
            订阅者对象，用 ``queue`` 取事件。
        """
        sub = _Subscriber(channels=channels)
        async with self._lock:
            self._subscribers.add(sub)
        _log.info("ws_subscribed", channels=sorted(channels), total=len(self._subscribers))
        try:
            yield sub
        finally:
            async with self._lock:
                self._subscribers.discard(sub)
            if sub.dropped:
                _log.warning("ws_events_dropped", dropped=sub.dropped)
            _log.info("ws_unsubscribed", total=len(self._subscribers))


def parse_channels(raw: str | None) -> frozenset[str]:
    """解析订阅参数。

    Args:
        raw: 逗号分隔的频道名；None 或空表示全订阅。

    Returns:
        合法频道集合。非法名直接忽略——一个拼错的频道名不该让整条连接建不起来。
    """
    if not raw or not raw.strip():
        return frozenset(CHANNELS)
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    valid = wanted & set(CHANNELS)
    if wanted - valid:
        _log.warning("ws_unknown_channels", unknown=sorted(wanted - valid))
    return frozenset(valid or CHANNELS)
