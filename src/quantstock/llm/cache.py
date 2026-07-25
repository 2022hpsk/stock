"""LLM 调用快照缓存（红线 LR3）。

规范见 docs/10-大模型集成规格.md 4.1。

```
cache_key = sha256(task_id + prompt_version + model_id + temperature + top_p
                   + canonical_json(input_payload))
```

**``canonical_json`` 必须真的规范化**：键排序、无多余空白、``ensure_ascii=False``。
否则同一份输入因为 dict 顺序不同就会算出两个 key，缓存命中率崩掉，
回测要么重跑要么退化成实时调用——后者在回测里是直接抛异常的。

``prompt_version`` 与 ``model_id`` 进入 key，也进入 ``param_hash``：
**改提示词等同于改策略**，不能让"只是润色了一下措辞"悄悄改变历史回测结果。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantstock.infra.clock import now
from quantstock.infra.logging import get_logger

__all__ = [
    "CacheEntry",
    "CacheStats",
    "LLMCache",
    "canonical_json",
    "compute_cache_key",
]

_log = get_logger(__name__)

SCHEMA_VERSION = 1


def canonical_json(payload: Any) -> str:  # noqa: ANN401 - 任意可 JSON 化的输入
    """规范化 JSON 序列化。

    键排序 + 紧凑分隔符 + 不转义非 ASCII。三者缺一都会让同一份逻辑输入
    算出不同的 key。

    Args:
        payload: 待序列化的对象。

    Returns:
        规范化字符串。
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_cache_key(
    *,
    task_id: str,
    prompt_version: str,
    model_id: str,
    temperature: float,
    payload: Any,  # noqa: ANN401 - 任务输入结构各异
    top_p: float = 1.0,
) -> str:
    """计算缓存键。

    Args:
        task_id: 任务标识。
        prompt_version: 提示词版本。
        model_id: 模型 ID。
        temperature: 采样温度。
        payload: 输入负载。
        top_p: 核采样参数。

    Returns:
        十六进制 SHA256。
    """
    material = canonical_json(
        {
            "task_id": task_id,
            "prompt_version": prompt_version,
            "model_id": model_id,
            "temperature": temperature,
            "top_p": top_p,
            "payload": payload,
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """一条缓存快照。

    存请求也存响应：只存响应的话，日后想搞清楚"当时到底问了什么"
    就只剩一个哈希值可看，排查幻觉与提示词退化都无从下手。
    """

    cache_key: str
    task_id: str
    model_id: str
    prompt_version: str
    temperature: float
    request: dict[str, Any]
    response: dict[str, Any]
    created_at: dt.datetime
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        """转成可落盘的字典。

        Returns:
            JSON 结构。
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "cache_key": self.cache_key,
            "task_id": self.task_id,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "request": self.request,
            "response": self.response,
            "created_at": self.created_at.isoformat(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CacheEntry:
        """从字典恢复。

        Args:
            payload: JSON 结构。

        Returns:
            缓存条目。
        """
        return cls(
            cache_key=str(payload["cache_key"]),
            task_id=str(payload["task_id"]),
            model_id=str(payload["model_id"]),
            prompt_version=str(payload["prompt_version"]),
            temperature=float(payload["temperature"]),
            request=dict(payload["request"]),
            response=dict(payload["response"]),
            created_at=dt.datetime.fromisoformat(str(payload["created_at"])),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
            latency_ms=int(payload.get("latency_ms", 0)),
        )


@dataclass
class CacheStats:
    """命中统计。

    回测跑完要看的第一个数字就是命中率：命中率低意味着大量决策点根本没有
    LLM 输出，此时"含 LLM 的 A/B 回测"实际上在比较两条几乎相同的路径。
    """

    hits: int = 0
    misses: int = 0
    writes: int = 0
    by_task: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """总查询次数。"""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """命中率。无查询时为 0。"""
        return self.hits / self.total if self.total else 0.0

    def record_hit(self, task_id: str) -> None:
        """记一次命中。

        Args:
            task_id: 任务标识。
        """
        self.hits += 1
        self.by_task[task_id] = self.by_task.get(task_id, 0) + 1

    def record_miss(self) -> None:
        """记一次未命中。"""
        self.misses += 1

    def message(self) -> str:
        """人类可读摘要。

        Returns:
            摘要文本。
        """
        return f"缓存命中 {self.hits}/{self.total}（{self.hit_rate:.1%}），写入 {self.writes}"


class LLMCache:
    """快照缓存。

    落盘布局 ``<root>/<task_id>/<key[:2]>/<key>.json``。
    两级目录是为了避免单目录塞进几万个文件——三年回测的 L1 缓存
    很容易到这个量级，某些文件系统在那之后会显著变慢。
    """

    def __init__(self, root: Path) -> None:
        """初始化。

        Args:
            root: 缓存根目录，通常是 ``var/llm_cache``。
        """
        self._root = Path(root)
        self.stats = CacheStats()

    @property
    def root(self) -> Path:
        """缓存根目录。"""
        return self._root

    def path_for(self, task_id: str, cache_key: str) -> Path:
        """缓存文件路径。

        Args:
            task_id: 任务标识。
            cache_key: 缓存键。

        Returns:
            文件路径。
        """
        return self._root / task_id / cache_key[:2] / f"{cache_key}.json"

    def get(self, task_id: str, cache_key: str) -> CacheEntry | None:
        """读取缓存。

        Args:
            task_id: 任务标识。
            cache_key: 缓存键。

        Returns:
            缓存条目；未命中或文件损坏时 None。
        """
        path = self.path_for(task_id, cache_key)
        if not path.exists():
            self.stats.record_miss()
            return None
        try:
            entry = CacheEntry.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            # 损坏的缓存按未命中处理而不是抛错：回放模式下未命中会被安全降级，
            # 抛错则会中断整个回测
            _log.warning("llm_cache_corrupt", path=str(path), error=str(exc))
            self.stats.record_miss()
            return None
        self.stats.record_hit(task_id)
        return entry

    def put(self, entry: CacheEntry) -> Path:
        """写入缓存。

        Args:
            entry: 缓存条目。

        Returns:
            写入的路径。
        """
        path = self.path_for(entry.task_id, entry.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.stats.writes += 1
        return path

    def has(self, task_id: str, cache_key: str) -> bool:
        """是否存在缓存（不计入命中统计）。

        Args:
            task_id: 任务标识。
            cache_key: 缓存键。

        Returns:
            存在则 True。
        """
        return self.path_for(task_id, cache_key).exists()

    def iter_entries(self, task_id: str | None = None) -> Iterator[CacheEntry]:
        """遍历缓存条目。

        Args:
            task_id: 只遍历该任务；None 表示全部。

        Yields:
            缓存条目。
        """
        base = self._root / task_id if task_id else self._root
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*.json")):
            try:
                yield CacheEntry.from_json(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                _log.warning("llm_cache_corrupt", path=str(path))

    def coverage(self, task_id: str, keys: list[str]) -> float:
        """给定一批键的覆盖率。

        回测前用它检查"这段区间的缓存备齐了没有"，避免跑到一半才发现
        大面积未命中。

        Args:
            task_id: 任务标识。
            keys: 待检查的缓存键。

        Returns:
            覆盖率 0~1；键为空时返回 1.0。
        """
        if not keys:
            return 1.0
        return sum(1 for k in keys if self.has(task_id, k)) / len(keys)

    def total_cost(self, task_id: str | None = None) -> float:
        """累计费用。

        Args:
            task_id: 只统计该任务；None 表示全部。

        Returns:
            美元金额。
        """
        return sum(e.cost_usd for e in self.iter_entries(task_id))

    def count(self, task_id: str | None = None) -> int:
        """条目数。

        Args:
            task_id: 只统计该任务；None 表示全部。

        Returns:
            条目数。
        """
        return sum(1 for _ in self.iter_entries(task_id))

    def make_entry(
        self,
        *,
        cache_key: str,
        task_id: str,
        model_id: str,
        prompt_version: str,
        temperature: float,
        request: dict[str, Any],
        response: dict[str, Any],
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
    ) -> CacheEntry:
        """构造一条缓存条目（时间取自注入的时钟）。

        Args:
            cache_key: 缓存键。
            task_id: 任务标识。
            model_id: 模型 ID。
            prompt_version: 提示词版本。
            temperature: 采样温度。
            request: 请求负载。
            response: 响应负载。
            input_tokens: 输入 token 数。
            output_tokens: 输出 token 数。
            cost_usd: 费用。
            latency_ms: 耗时。

        Returns:
            缓存条目。
        """
        return CacheEntry(
            cache_key=cache_key,
            task_id=task_id,
            model_id=model_id,
            prompt_version=prompt_version,
            temperature=temperature,
            request=request,
            response=response,
            created_at=now(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
