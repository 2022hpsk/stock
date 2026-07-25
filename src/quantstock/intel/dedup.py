"""情报去重。

规范见 docs/07-信息情报模块.md 4.2。

两层：

1. **精确去重**：``content_hash``（标题+正文归一化后 SHA256）。同一条消息被多个源
   一字不差地转载时命中。
2. **近似去重**：64 位 SimHash。财经快讯的典型形态是同一事件被十几家媒体改写标题转发，
   精确哈希完全拦不住，而它们会让 importance 的"多源印证"项虚高。

**SimHash 自己实现**而不是引入依赖：核心算法不到 40 行，而 ``simhash`` 这类包
为了通用性会引入分词器等重量级依赖。中文场景下我们用字符 n-gram 而非分词——
财经文本里的实体（公司简称、政策名）常被分词器切碎，n-gram 反而更稳。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from quantstock.intel.types import IntelItem

__all__ = [
    "SIMHASH_BITS",
    "DedupResult",
    "content_hash",
    "dedup",
    "hamming_distance",
    "normalize_text",
    "simhash",
    "similarity",
]

SIMHASH_BITS = 64

DEFAULT_NGRAM = 4
"""字符 n-gram 长度。

取 4 是量出来的，不是拍的。在一组真实形态的财经标题上（同一事件的媒体改写
对 vs 无关新闻对），三种 gram 长度的可分性是：

===========  ==============  ==============  ======
gram 长度    改写对最低相似  无关对最高相似  间隔
===========  ==============  ==============  ======
2            0.750           0.609           0.141
3            0.641           0.516           0.125
4            0.766           0.547           0.219
===========  ==============  ==============  ======

4-gram 的间隔最宽。直觉上也说得通：中文财经标题里"控股股东"、"立案调查"、
"同比上涨"这类四字组合本身就是语义单元，切成 4-gram 恰好把它们保留成一个特征。
"""

DEFAULT_SIMILARITY_THRESHOLD = 0.65
"""近似去重阈值，取在上表 4-gram 那一行的间隔中点附近。

**这个值不能想当然地设成 0.9**。SimHash 是 64 位有损指纹，不是文本相似度：
两条只改了标点和几个虚词的标题，指纹仍会差十几个比特。实测同一事件的媒体改写
落在 0.64~0.77，0.9 的阈值等于近似去重根本不会触发——精确哈希漏掉的转载
会全部当成独立事件，直接把 importance 的"多源印证"项灌成噪声。
"""

DEFAULT_TIME_WINDOW_HOURS = 6

_PUNCT = re.compile(r"[\s　!-/:-@\[-`{-~！-／：-＠［-｀｛-～、。「」『』【】—…·]+")


def normalize_text(text: str) -> str:
    """文本归一化。

    全角转半角、去标点空白、统一小写。做这一步是因为同一条快讯在不同源上
    常常只差全半角标点与空格。

    Args:
        text: 原始文本。

    Returns:
        归一化结果。
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _PUNCT.sub("", folded)


def content_hash(title: str, body: str = "") -> str:
    """内容指纹，用于精确去重。

    Args:
        title: 标题。
        body: 正文。

    Returns:
        十六进制 SHA256。
    """
    payload = normalize_text(title) + "\x00" + normalize_text(body)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shingles(text: str, n: int = DEFAULT_NGRAM) -> Counter[str]:
    """切成字符 n-gram 并计数。

    Args:
        text: 归一化后的文本。
        n: gram 长度。

    Returns:
        gram → 出现次数。
    """
    if len(text) <= n:
        return Counter([text] if text else [])
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def simhash(text: str, *, ngram: int = DEFAULT_NGRAM) -> int:
    """计算 64 位 SimHash。

    做法：把每个 n-gram 的 SHA256 取低 64 位作为特征向量，按出现次数加权，
    逐位累加（置位 +w、清位 -w），最后取符号位。相似文本的多数比特会一致。

    Args:
        text: 原始文本，内部会先归一化。
        ngram: gram 长度。

    Returns:
        64 位无符号整数。
    """
    normalized = normalize_text(text)
    if not normalized:
        return 0

    weights = [0] * SIMHASH_BITS
    for gram, count in _shingles(normalized, ngram).items():
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        for bit in range(SIMHASH_BITS):
            if value >> bit & 1:
                weights[bit] += count
            else:
                weights[bit] -= count

    result = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    """两个指纹的汉明距离。

    Args:
        left: 指纹一。
        right: 指纹二。

    Returns:
        不同比特数，0~64。
    """
    return (left ^ right).bit_count()


def similarity(left: int, right: int) -> float:
    """由汉明距离换算的相似度。

    Args:
        left: 指纹一。
        right: 指纹二。

    Returns:
        0~1，1 表示指纹完全相同。
    """
    return 1.0 - hamming_distance(left, right) / SIMHASH_BITS


@dataclass(frozen=True, slots=True)
class DedupResult:
    """去重结果。"""

    kept: tuple[IntelItem, ...]
    """保留的主条目，已把重复项 id 记入 ``duplicates``。"""
    dropped: tuple[IntelItem, ...]
    """被合并掉的条目。保留下来是为了可追溯，不进入后续流水线。"""

    @property
    def dropped_count(self) -> int:
        """合并掉的条数。"""
        return len(self.dropped)


def _pick_primary(left: IntelItem, right: IntelItem) -> tuple[IntelItem, IntelItem]:
    """在两个重复条目里选主条目。

    按 ``source_tier`` 高者优先——交易所公告永远压过媒体转述；
    同级时取发布更早者，因为首发比转载更接近事实源头。

    Args:
        left: 条目一。
        right: 条目二。

    Returns:
        ``(主条目, 被合并条目)``。
    """
    if left.source_tier != right.source_tier:
        return (left, right) if left.source_tier > right.source_tier else (right, left)
    return (left, right) if left.publish_at <= right.publish_at else (right, left)


def dedup(
    items: Iterable[IntelItem],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    window_hours: int = DEFAULT_TIME_WINDOW_HOURS,
) -> DedupResult:
    """对一批情报去重。

    近似去重只在**时间窗内**比较：同一措辞的消息隔了三天再出现，通常是
    "旧闻重发"或"事件进展"，不该被当成重复合并掉。

    Args:
        items: 待去重的条目。
        threshold: 相似度阈值，超过即视为同一事件。
        window_hours: 时间窗，单位小时。

    Returns:
        去重结果。
    """
    ordered = sorted(items, key=lambda i: (i.publish_at, i.item_id))
    fingerprints: list[tuple[IntelItem, int]] = []
    merged: dict[str, list[str]] = {}
    dropped: list[IntelItem] = []
    exact: dict[str, int] = {}
    window = window_hours * 3600

    for item in ordered:
        primary_index = exact.get(item.content_hash)
        if primary_index is None:
            primary_index = _find_near_duplicate(item, fingerprints, threshold, window)

        if primary_index is None:
            exact[item.content_hash] = len(fingerprints)
            fingerprints.append((item, simhash(f"{item.title}\n{item.body}")))
            continue

        primary = fingerprints[primary_index][0]
        winner, loser = _pick_primary(primary, item)
        merged.setdefault(winner.item_id, []).extend(
            [loser.item_id, *merged.pop(loser.item_id, [])]
        )
        dropped.append(loser)
        if winner is not primary:
            fingerprints[primary_index] = (winner, simhash(f"{winner.title}\n{winner.body}"))
            exact[winner.content_hash] = primary_index

    kept = tuple(_attach_duplicates(item, merged.get(item.item_id, ())) for item, _ in fingerprints)
    return DedupResult(kept=kept, dropped=tuple(dropped))


def _find_near_duplicate(
    item: IntelItem,
    fingerprints: Sequence[tuple[IntelItem, int]],
    threshold: float,
    window_seconds: int,
) -> int | None:
    """在已有条目里找近似重复。

    Args:
        item: 待判定条目。
        fingerprints: 已保留条目及其指纹。
        threshold: 相似度阈值。
        window_seconds: 时间窗秒数。

    Returns:
        命中的索引；无命中则 None。
    """
    probe = simhash(f"{item.title}\n{item.body}")
    for index, (existing, existing_print) in enumerate(fingerprints):
        gap = abs((item.publish_at - existing.publish_at).total_seconds())
        if gap > window_seconds:
            continue
        if similarity(probe, existing_print) >= threshold:
            return index
    return None


def _attach_duplicates(item: IntelItem, duplicates: Sequence[str]) -> IntelItem:
    """把重复条目 id 挂到主条目上。

    Args:
        item: 主条目。
        duplicates: 被合并的 id。

    Returns:
        带 ``duplicates`` 的新条目。
    """
    if not duplicates:
        return item
    from dataclasses import replace  # noqa: PLC0415 - 仅此处需要

    return replace(item, duplicates=tuple(dict.fromkeys(duplicates)))
