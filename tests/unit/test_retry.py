"""重试与限流测试。

测试必须确定性：休眠函数与随机源全部注入，不产生真实等待。
"""

from __future__ import annotations

import random

import pytest

from quantstock.infra.retry import RateLimiter, RetryPolicy, chunked, retry


class FakeSleep:
    """记录休眠调用而不真正等待。"""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.calls)


class FakeClock:
    """可手动推进的单调时钟。"""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TestRetryPolicy:
    @pytest.mark.parametrize(
        ("attempt", "expected"),
        [(1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0)],
    )
    def test_delay_for__exponential_without_jitter(self, attempt: int, expected: float) -> None:
        policy = RetryPolicy(base_delay=1.0, multiplier=2.0, jitter=0.0)
        assert policy.delay_for(attempt) == expected

    def test_delay_for__capped_at_max(self) -> None:
        policy = RetryPolicy(base_delay=1.0, multiplier=10.0, max_delay=5.0, jitter=0.0)
        assert policy.delay_for(10) == 5.0

    def test_delay_for__jitter_stays_in_band(self) -> None:
        policy = RetryPolicy(base_delay=10.0, jitter=0.2)
        rng = random.Random(42)
        for _ in range(50):
            delay = policy.delay_for(1, rng=rng)
            assert 8.0 <= delay <= 12.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_attempts": 0}, "max_attempts 必须"),
            ({"base_delay": -1.0}, "退避时间不能为负"),
            ({"jitter": 1.5}, "jitter 必须"),
        ],
    )
    def test_invalid_params__raise(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            RetryPolicy(**kwargs)  # type: ignore[arg-type]


class TestRetry:
    def test_succeeds_first_try__no_sleep(self) -> None:
        sleep = FakeSleep()
        calls = 0

        @retry(policy=RetryPolicy(jitter=0.0), sleep=sleep)
        def works() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        assert works() == "ok"
        assert calls == 1
        assert sleep.calls == []

    def test_retries_then_succeeds(self) -> None:
        sleep = FakeSleep()
        attempts = 0

        @retry(policy=RetryPolicy(max_attempts=3, base_delay=1.0, jitter=0.0), sleep=sleep)
        def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                msg = "transient"
                raise ConnectionError(msg)
            return "ok"

        assert flaky() == "ok"
        assert attempts == 3
        assert sleep.calls == [1.0, 2.0]

    def test_exhausts_attempts__reraises_last_error(self) -> None:
        sleep = FakeSleep()
        attempts = 0

        @retry(policy=RetryPolicy(max_attempts=3, jitter=0.0), sleep=sleep)
        def always_fails() -> None:
            nonlocal attempts
            attempts += 1
            msg = f"failure #{attempts}"
            raise ConnectionError(msg)

        with pytest.raises(ConnectionError, match="failure #3"):
            always_fails()
        assert attempts == 3

    def test_unlisted_exception__not_retried(self) -> None:
        """只重试指定异常，其它异常必须立刻上抛——尤其是资金相关的错误。"""
        sleep = FakeSleep()
        attempts = 0

        @retry(on=ConnectionError, policy=RetryPolicy(max_attempts=5), sleep=sleep)
        def raises_value_error() -> None:
            nonlocal attempts
            attempts += 1
            msg = "not retryable"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="not retryable"):
            raises_value_error()
        assert attempts == 1
        assert sleep.calls == []

    def test_preserves_function_metadata(self) -> None:
        @retry()
        def documented() -> None:
            """Some docstring."""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Some docstring."


class TestRateLimiter:
    def test_try_acquire__within_burst__succeeds(self) -> None:
        limiter = RateLimiter(rate_per_min=60, burst=5, monotonic=FakeClock(), sleep=FakeSleep())
        assert all(limiter.try_acquire() for _ in range(5))

    def test_try_acquire__exhausted__fails(self) -> None:
        limiter = RateLimiter(rate_per_min=60, burst=2, monotonic=FakeClock(), sleep=FakeSleep())
        assert limiter.try_acquire()
        assert limiter.try_acquire()
        assert not limiter.try_acquire()

    def test_refill_over_time(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(rate_per_min=60, burst=1, monotonic=clock, sleep=FakeSleep())
        assert limiter.try_acquire()
        assert not limiter.try_acquire()
        clock.advance(1.0)  # 60/min = 1 token/sec
        assert limiter.try_acquire()

    def test_acquire__blocks_until_available(self) -> None:
        clock = FakeClock()
        sleep = FakeSleep()

        def advancing_sleep(seconds: float) -> None:
            sleep(seconds)
            clock.advance(seconds)

        limiter = RateLimiter(rate_per_min=60, burst=1, monotonic=clock, sleep=advancing_sleep)
        limiter.acquire()
        waited = limiter.acquire()
        assert waited == pytest.approx(1.0)

    def test_acquire__more_than_capacity__raises(self) -> None:
        limiter = RateLimiter(rate_per_min=60, burst=2, monotonic=FakeClock(), sleep=FakeSleep())
        with pytest.raises(ValueError, match="超过桶容量"):
            limiter.acquire(tokens=5)

    def test_non_positive_rate__raises(self) -> None:
        with pytest.raises(ValueError, match="rate_per_min 必须为正"):
            RateLimiter(rate_per_min=0)


class TestChunked:
    def test_splits_evenly(self) -> None:
        assert list(chunked(range(6), 2)) == [[0, 1], [2, 3], [4, 5]]

    def test_last_batch_partial(self) -> None:
        assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]

    def test_empty_input(self) -> None:
        assert list(chunked([], 3)) == []

    def test_invalid_size__raises(self) -> None:
        with pytest.raises(ValueError, match="size 必须为正整数"):
            list(chunked([1, 2], 0))
