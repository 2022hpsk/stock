"""重试与限流。

规范见 docs/01-开发规范.md 第六、十条：

- 重试统一走本模块（指数退避 + 抖动 + 上限），禁止手写 ``while True: sleep()``。
- 外部数据源调用必须限流。
- **涉及资金的操作不得重试**——下单失败必须 fail-fast，重试可能造成重复下单。
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import wraps
from typing import ParamSpec, TypeVar

from quantstock.infra.logging import get_logger

__all__ = ["RateLimiter", "RetryPolicy", "retry"]

P = ParamSpec("P")
T = TypeVar("T")

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """重试策略。

    Attributes:
        max_attempts: 最大尝试次数（含首次），必须 ≥ 1。
        base_delay: 首次退避秒数。
        max_delay: 单次退避上限秒数。
        multiplier: 退避倍数。
        jitter: 抖动比例，实际延迟为 ``delay × (1 ± jitter)``，避免多源同时重试造成雪崩。
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.2

    def __post_init__(self) -> None:
        """校验参数合法性。

        Raises:
            ValueError: 参数不合法。
        """
        if self.max_attempts < 1:
            msg = f"max_attempts 必须 ≥ 1，收到 {self.max_attempts}"
            raise ValueError(msg)
        if self.base_delay < 0 or self.max_delay < 0:
            msg = "退避时间不能为负"
            raise ValueError(msg)
        if not 0 <= self.jitter <= 1:
            msg = f"jitter 必须在 [0, 1] 之间，收到 {self.jitter}"
            raise ValueError(msg)

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """计算第 attempt 次失败后的退避秒数。

        Args:
            attempt: 已失败的次数，从 1 开始。
            rng: 随机源，测试时可注入以保证确定性。

        Returns:
            退避秒数，非负。
        """
        raw = min(self.base_delay * (self.multiplier ** (attempt - 1)), self.max_delay)
        if self.jitter == 0:
            return raw
        source = rng or random
        factor = 1 + source.uniform(-self.jitter, self.jitter)
        return max(0.0, raw * factor)


def retry(
    *,
    on: type[Exception] | tuple[type[Exception], ...] = Exception,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """带指数退避的重试装饰器。

    只用于**幂等**的读取操作（拉数据、查行情）。
    **禁止**用于下单、撤单等涉及资金的写操作。

    Args:
        on: 需要重试的异常类型。
        policy: 重试策略，默认 :class:`RetryPolicy`。
        sleep: 休眠函数，测试时可注入假实现避免真实等待。
        rng: 随机源，测试时可注入。

    Returns:
        装饰器。
    """
    effective = policy or RetryPolicy()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last: Exception | None = None
            for attempt in range(1, effective.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except on as exc:
                    last = exc
                    if attempt >= effective.max_attempts:
                        break
                    delay = effective.delay_for(attempt, rng=rng)
                    _log.warning(
                        "call_failed_retrying",
                        func=func.__qualname__,
                        attempt=attempt,
                        max_attempts=effective.max_attempts,
                        delay_sec=round(delay, 3),
                        error=str(exc),
                    )
                    sleep(delay)
            assert last is not None  # noqa: S101 - 循环必然经过至少一次 except
            _log.error(
                "call_failed_giving_up",
                func=func.__qualname__,
                attempts=effective.max_attempts,
                error=str(last),
            )
            raise last

        return wrapper

    return decorator


class RateLimiter:
    """令牌桶限流器，线程安全。

    用于对外部数据源的礼貌抓取（见 docs/07-信息情报模块.md I-R6）。

    Example:
        >>> limiter = RateLimiter(rate_per_min=60)
        >>> limiter.acquire()  # 需要时会阻塞到有令牌
    """

    def __init__(
        self,
        *,
        rate_per_min: float,
        burst: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """初始化。

        Args:
            rate_per_min: 每分钟允许的请求数，必须为正。
            burst: 桶容量（允许的突发请求数），默认与每分钟速率相同。
            monotonic: 单调时钟，测试时可注入。
            sleep: 休眠函数，测试时可注入。

        Raises:
            ValueError: rate_per_min 非正。
        """
        if rate_per_min <= 0:
            msg = f"rate_per_min 必须为正，收到 {rate_per_min}"
            raise ValueError(msg)
        self._rate_per_sec = rate_per_min / 60.0
        self._capacity = float(burst if burst is not None else max(1, int(rate_per_min)))
        self._tokens = self._capacity
        self._monotonic = monotonic
        self._sleep = sleep
        self._last = monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """按流逝时间补充令牌。调用方必须已持锁。"""
        current = self._monotonic()
        elapsed = current - self._last
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
            self._last = current

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """尝试获取令牌，不阻塞。

        Args:
            tokens: 需要的令牌数。

        Returns:
            成功获取则 True，令牌不足则 False。
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0) -> float:
        """获取令牌，必要时阻塞等待。

        Args:
            tokens: 需要的令牌数，不得超过桶容量。

        Returns:
            实际等待的秒数。

        Raises:
            ValueError: 请求令牌数超过桶容量（永远等不到）。
        """
        if tokens > self._capacity:
            msg = f"请求令牌数 {tokens} 超过桶容量 {self._capacity}，将永远无法满足"
            raise ValueError(msg)
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                wait = deficit / self._rate_per_sec
            self._sleep(wait)
            waited += wait


def chunked(items: Iterable[T], size: int) -> Iterable[list[T]]:
    """把可迭代对象切成定长批次。

    数据源批量接口常有单次条数上限，用本函数切批。

    Args:
        items: 待切分的元素。
        size: 每批大小，必须为正。

    Yields:
        每批元素组成的列表，最后一批可能不满。

    Raises:
        ValueError: size 非正。
    """
    if size <= 0:
        msg = f"size 必须为正整数，收到 {size}"
        raise ValueError(msg)
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
