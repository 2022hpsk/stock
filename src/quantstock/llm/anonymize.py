"""标的匿名化：防训练集泄漏的第二道防线（红线 LR4）。

规范见 docs/10-大模型集成规格.md 5.1。

**回测默认开启，实盘默认关闭。**

原理：模型的知识截止日期晚于回测区间。给它看 2023 年某公司的材料并问
"这些因素是正面还是负面"，如果它认得出这是哪家公司，它就知道后来发生了什么，
回测结果会好得离谱且完全虚假。把公司名换成"标的A"能显著削弱这种定位能力。

**这不是完美防御**：材料本身可能含有足以推断出主体的信息（"茅台镇"、
"动力电池龙头"）。它是第二道防线，第一道是任务边界——不问预测问题。
第三道是 5.2 的泄漏检测测试。

替换必须**稳定且可还原**：同一次调用里同一标的始终映射到同一代号，
否则模型看到的材料自相矛盾；还原表留在本地，输出里的代号要能换回真实标的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from quantstock.infra.types import Symbol

__all__ = ["AnonymizedText", "Anonymizer", "strip_absolute_dates"]

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_DATE_PATTERNS = (
    re.compile(r"\b(19|20)\d{2}[-/年]\s?\d{1,2}[-/月]\s?\d{1,2}\s?日?"),
    re.compile(r"\b(19|20)\d{2}\s?年"),
    re.compile(r"\b(19|20)\d{2}[-/]\d{1,2}\b"),
)


def strip_absolute_dates(text: str, placeholder: str = "〈日期〉") -> str:
    """把绝对日期换成占位符。

    回测提示词里出现"2023年3月15日"等于直接告诉模型现在是哪一天，
    它的先验知识立刻就能派上用场。改用相对表述后模型只能依据材料本身。

    Args:
        text: 原始文本。
        placeholder: 替换成的占位符。

    Returns:
        脱敏后的文本。
    """
    out = text
    for pattern in _DATE_PATTERNS:
        out = pattern.sub(placeholder, out)
    return out


@dataclass(frozen=True, slots=True)
class AnonymizedText:
    """匿名化结果。"""

    text: str
    mapping: dict[str, Symbol]
    """代号 → 真实标的。用于把模型输出里的代号还原回去。"""

    @property
    def is_anonymized(self) -> bool:
        """是否实际发生了替换。"""
        return bool(self.mapping)


@dataclass
class Anonymizer:
    """标的匿名化器。

    一次决策周期用一个实例：同一标的在整批材料里必须是同一个代号。
    """

    prefix: str = "标的"
    _forward: dict[Symbol, str] = field(default_factory=dict, init=False)
    _names: dict[str, Symbol] = field(default_factory=dict, init=False)

    def register(self, symbol: Symbol, *names: str) -> str:
        """登记一个标的及其别名，返回代号。

        Args:
            symbol: 标的。
            *names: 公司名、简称、曾用名。

        Returns:
            分配的代号。
        """
        label = self._forward.get(symbol)
        if label is None:
            label = f"{self.prefix}{self._next_label()}"
            self._forward[symbol] = label

        self._names[str(symbol)] = symbol
        code = str(symbol).split(".")[0]
        self._names[code] = symbol
        for name in names:
            if name.strip():
                self._names[name.strip()] = symbol
        return label

    def _next_label(self) -> str:
        """生成下一个代号后缀：A…Z，然后 AA、AB……

        Returns:
            代号后缀。
        """
        index = len(self._forward)
        if index < len(_LABELS):
            return _LABELS[index]
        first, second = divmod(index - len(_LABELS), len(_LABELS))
        return _LABELS[first] + _LABELS[second]

    def label_of(self, symbol: Symbol) -> str:
        """取某标的的代号，未登记时自动登记。

        Args:
            symbol: 标的。

        Returns:
            代号。
        """
        return self._forward.get(symbol) or self.register(symbol)

    def anonymize(self, text: str) -> AnonymizedText:
        """替换文本中出现的标的名与代码。

        长名优先替换，避免"中国平安"被"平安"抢先命中后留下"中国标的A"。

        Args:
            text: 原始文本。

        Returns:
            匿名化结果。
        """
        out = text
        used: dict[str, Symbol] = {}
        for name in sorted(self._names, key=len, reverse=True):
            if name not in out:
                continue
            symbol = self._names[name]
            label = self.label_of(symbol)
            out = out.replace(name, label)
            used[label] = symbol
        return AnonymizedText(text=out, mapping=used)

    def restore(self, text: str) -> str:
        """把代号换回真实标的。

        Args:
            text: 含代号的文本。

        Returns:
            还原后的文本。
        """
        out = text
        for symbol, label in self._forward.items():
            out = out.replace(label, str(symbol))
        return out

    def resolve(self, label: str) -> Symbol | None:
        """由代号找回标的。

        Args:
            label: 代号，如 ``标的A``。

        Returns:
            标的；未登记时 None。
        """
        for symbol, mapped in self._forward.items():
            if mapped == label:
                return symbol
        return None

    @property
    def mapping(self) -> dict[Symbol, str]:
        """标的 → 代号的完整映射，供审计。"""
        return dict(self._forward)
