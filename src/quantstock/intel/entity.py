"""实体链接：把情报文本挂到标的、行业与主题上。

规范见 docs/07-信息情报模块.md 4.3。

**匹配结果必须可解释**——每次命中都记录命中的关键词，日报里才能说清楚
"这条消息为什么被认为跟贵州茅台有关"。不可解释的关联比没有关联更糟：
它会让人误以为系统看到了它其实没看到的东西。

歧义消解优先当前持仓与候选池：同一个简称可能对应多家公司，
但用户真正关心的是自己持有的那一家。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from quantstock.infra.types import Symbol, parse_symbol

__all__ = ["EntityLinker", "LinkResult", "SymbolDictionary"]

_CODE_PATTERN = re.compile(r"\b(\d{6})(?:\.(?:SH|SZ|BJ|sh|sz|bj))?\b")
MIN_ALIAS_LENGTH = 2
"""短于 2 个字的别名不参与匹配——单字简称的误报率高到没有使用价值。"""


@dataclass(frozen=True, slots=True)
class SymbolDictionary:
    """标的词典。

    ``aliases`` 含简称、曾用名、简称变体。曾用名必须保留：
    公司改名后旧闻仍在流传，丢掉曾用名会让历史情报回补大面积漏匹配。
    """

    symbol: Symbol
    name: str
    aliases: tuple[str, ...] = ()
    industry: str = ""

    def all_names(self) -> tuple[str, ...]:
        """全部可匹配名称，按长度降序。

        长名优先匹配，避免"中国平安"被"平安"抢先命中。

        Returns:
            名称元组。
        """
        names = {self.name, *self.aliases}
        long_enough = (n for n in names if len(n) >= MIN_ALIAS_LENGTH)
        return tuple(sorted(long_enough, key=len, reverse=True))


@dataclass(frozen=True, slots=True)
class LinkResult:
    """一次实体链接的结果。"""

    symbols: tuple[Symbol, ...]
    industries: tuple[str, ...]
    themes: tuple[str, ...]
    evidence: tuple[str, ...]
    """命中的关键词，形如 ``600519.SH←贵州茅台``。"""

    @property
    def is_empty(self) -> bool:
        """是否什么都没匹配到。"""
        return not (self.symbols or self.industries or self.themes)


class EntityLinker:
    """实体链接器。"""

    def __init__(
        self,
        symbols: Iterable[SymbolDictionary] = (),
        *,
        industries: Mapping[str, Sequence[str]] | None = None,
        themes: Mapping[str, Sequence[str]] | None = None,
        priority: Iterable[Symbol] = (),
    ) -> None:
        """初始化。

        Args:
            symbols: 标的词典。
            industries: 行业名 → 关键词列表。
            themes: 主题名 → 关键词列表。
            priority: 优先标的（当前持仓 + 候选池），用于歧义消解。
        """
        self._dicts = tuple(symbols)
        self._by_code = {str(d.symbol).split(".")[0]: d for d in self._dicts}
        self._industries = {k: tuple(v) for k, v in (industries or {}).items()}
        self._themes = {k: tuple(v) for k, v in (themes or {}).items()}
        self._priority = frozenset(priority)

        # 别名 → 候选标的。同名多主体时保留全部，在消解阶段再选
        alias_map: dict[str, list[SymbolDictionary]] = {}
        for entry in self._dicts:
            for name in entry.all_names():
                alias_map.setdefault(name, []).append(entry)
        # 长别名优先，避免短名抢匹配
        self._aliases = dict(sorted(alias_map.items(), key=lambda kv: len(kv[0]), reverse=True))

    def link(self, title: str, body: str = "") -> LinkResult:
        """对一条情报做实体链接。

        Args:
            title: 标题。
            body: 正文。

        Returns:
            链接结果。
        """
        text = f"{title}\n{body}"
        symbols: dict[Symbol, None] = {}
        evidence: list[str] = []

        for code in _CODE_PATTERN.findall(text):
            entry = self._by_code.get(code)
            if entry is not None:
                symbols.setdefault(entry.symbol)
                evidence.append(f"{entry.symbol}←代码{code}")
            else:
                # 词典里没有也照样归一化：universe 可能落后于新股上市
                try:
                    symbols.setdefault(parse_symbol(code))
                    evidence.append(f"{code}←代码")
                except ValueError:
                    continue  # 非法代码跳过即可

        for alias, candidates in self._aliases.items():
            if alias not in text:
                continue
            chosen = self._disambiguate(candidates)
            if chosen.symbol in symbols:
                continue
            symbols[chosen.symbol] = None
            evidence.append(f"{chosen.symbol}←{alias}")

        industries = self._match_keywords(text, self._industries)
        industries.extend(
            entry.industry
            for entry in self._dicts
            if entry.symbol in symbols and entry.industry and entry.industry not in industries
        )
        themes = self._match_keywords(text, self._themes)

        return LinkResult(
            symbols=tuple(symbols),
            industries=tuple(dict.fromkeys(industries)),
            themes=tuple(themes),
            evidence=tuple(evidence),
        )

    def _disambiguate(self, candidates: Sequence[SymbolDictionary]) -> SymbolDictionary:
        """同名多主体时选一个。

        Args:
            candidates: 候选标的。

        Returns:
            选中的标的。持仓/候选池内的优先。
        """
        if len(candidates) == 1:
            return candidates[0]
        for candidate in candidates:
            if candidate.symbol in self._priority:
                return candidate
        return candidates[0]

    @staticmethod
    def _match_keywords(text: str, table: Mapping[str, tuple[str, ...]]) -> list[str]:
        """按关键词表匹配标签。

        Args:
            text: 待匹配文本。
            table: 标签 → 关键词。

        Returns:
            命中的标签。
        """
        return [label for label, words in table.items() if any(w in text for w in words)]
