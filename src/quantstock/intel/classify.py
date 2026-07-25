"""事件分类与情绪打分。

规范见 docs/07-信息情报模块.md 4.4、4.6。

**两者都以规则为主、LLM 为辅**，理由是一样的：确定性高、可回测、可解释。
一条"立案调查"在 2019 年和 2026 年必须被分到同一类，否则历史情报回补出来的
因子序列会随模型版本漂移，回测结果就没有意义了。

LLM 只在规则完全未命中时兜底，且结果标注 ``classifier="llm:<model>"``（红线 I-R3）。
本模块**不 import ``llm``**——兜底分类由上层 pipeline 注入回调，
以免情报层对模型层产生编译期依赖。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quantstock.intel.types import EventType, IntelDomain

__all__ = [
    "EVENT_KEYWORDS",
    "EVENT_SENTIMENT_PRIOR",
    "ClassifyResult",
    "EventClassifier",
    "SentimentScorer",
]

EVENT_KEYWORDS: Mapping[EventType, tuple[str, ...]] = {
    # ---- 风险类放最前：命中即优先，宁可把中性消息误判为风险，也不要漏掉风险 ----
    EventType.REGULATORY_PROBE: (
        "立案调查",
        "立案侦查",
        "被调查",
        "行政处罚",
        "责令改正",
        "监管函",
        "警示函",
    ),
    EventType.AUDIT_QUALIFIED: ("非标准审计", "保留意见", "无法表示意见", "否定意见", "内控否定"),
    EventType.DELISTING_RISK: ("退市风险", "终止上市", "面值退市", "暂停上市", "退市整理"),
    EventType.CONTROL_CHANGE: ("实控人变更", "实际控制人变更", "控制权变更", "易主"),
    EventType.GUARANTEE_RISK: ("违规担保", "对外担保逾期", "担保代偿"),
    EventType.PLEDGE_RISK: ("质押平仓", "股份被冻结", "质押比例", "补充质押"),
    EventType.LITIGATION: ("诉讼", "仲裁", "起诉", "法院受理", "败诉"),
    # ---- 业绩类 ----
    EventType.EARNINGS_FORECAST: ("业绩预告", "业绩预增", "业绩预减", "预亏", "预盈", "业绩快报"),
    EventType.EARNINGS_SURPRISE: ("超预期", "大超预期", "低于预期", "不及预期"),
    EventType.EARNINGS_REPORT: ("年报", "半年报", "季报", "财报", "营业收入", "归母净利润"),
    # ---- 股权类 ----
    EventType.SHAREHOLDER_REDUCE: ("减持", "拟减持", "清仓式减持", "套现"),
    EventType.SHAREHOLDER_INCREASE: ("增持", "拟增持", "举牌"),
    EventType.BUYBACK: ("回购", "股份回购", "回购注销"),
    EventType.PLACEMENT: ("定增", "定向增发", "非公开发行", "可转债发行", "配股"),
    EventType.UNLOCK: ("解禁", "限售股上市", "限售解除"),
    # ---- 经营类 ----
    EventType.MAJOR_CONTRACT: ("中标", "大额订单", "重大合同", "签署战略合作", "框架协议"),
    EventType.MA: ("并购", "重组", "收购", "吸收合并", "借壳"),
    EventType.ASSET_SALE: ("出售资产", "资产处置", "剥离"),
    EventType.CAPACITY: ("投产", "扩产", "产能", "停产", "检修"),
    # ---- 交易类 ----
    EventType.SUSPENSION: ("停牌", "复牌"),
    EventType.ST_CHANGE: ("*ST", "ST股", "撤销风险警示", "实施其他风险警示"),
    EventType.INDEX_ADJUST: ("指数调整", "纳入指数", "调出指数", "样本股调整"),
    # ---- 分红类 ----
    EventType.DIVIDEND: ("分红", "派息", "现金分红", "利润分配"),
    EventType.SPLIT: ("送转", "送股", "转增"),
    # ---- 行业政策 ----
    EventType.POLICY_NEGATIVE: ("收紧", "限制", "禁止", "整顿", "从严", "提高门槛"),
    EventType.POLICY_POSITIVE: ("扶持", "补贴", "减税", "支持政策", "鼓励", "试点放开"),
    EventType.PRICE_MOVE: ("涨价", "提价", "降价", "报价上调", "价格下调"),
    # ---- 宏观 ----
    EventType.MONETARY: ("降准", "降息", "加息", "MLF", "LPR", "公开市场操作", "逆回购"),
    EventType.FISCAL: ("专项债", "财政赤字", "减税降费", "特别国债"),
    EventType.MACRO_DATA: ("CPI", "PPI", "PMI", "GDP", "社融", "进出口数据", "失业率"),
    EventType.GEOPOLITICS: ("制裁", "关税", "地缘", "冲突", "出口管制"),
}
"""事件关键词表。字典顺序即优先级——风险类在最前。

Python 3.7+ 的 dict 保序，这里刻意依赖这一点：一条既提到"重组"又提到"立案调查"的
公告应该被分为风险类。
"""

EVENT_SENTIMENT_PRIOR: Mapping[EventType, float] = {
    EventType.REGULATORY_PROBE: -0.9,
    EventType.AUDIT_QUALIFIED: -0.85,
    EventType.DELISTING_RISK: -0.95,
    EventType.CONTROL_CHANGE: -0.4,
    EventType.GUARANTEE_RISK: -0.7,
    EventType.PLEDGE_RISK: -0.5,
    EventType.LITIGATION: -0.35,
    EventType.EARNINGS_FORECAST: 0.0,
    EventType.EARNINGS_REPORT: 0.0,
    EventType.EARNINGS_SURPRISE: 0.0,
    EventType.SHAREHOLDER_REDUCE: -0.45,
    EventType.SHAREHOLDER_INCREASE: 0.45,
    EventType.BUYBACK: 0.4,
    EventType.PLACEMENT: -0.15,
    EventType.UNLOCK: -0.3,
    EventType.MAJOR_CONTRACT: 0.5,
    EventType.MA: 0.25,
    EventType.ASSET_SALE: 0.05,
    EventType.CAPACITY: 0.1,
    EventType.SUSPENSION: -0.1,
    EventType.ST_CHANGE: -0.6,
    EventType.INDEX_ADJUST: 0.1,
    EventType.DIVIDEND: 0.3,
    EventType.SPLIT: 0.15,
    EventType.POLICY_POSITIVE: 0.5,
    EventType.POLICY_NEGATIVE: -0.5,
    EventType.PRICE_MOVE: 0.0,
    EventType.MONETARY: 0.0,
    EventType.FISCAL: 0.2,
    EventType.MACRO_DATA: 0.0,
    EventType.GEOPOLITICS: -0.3,
}
"""事件类型的情绪先验。

几处刻意取 0 而非猜一个方向：``EARNINGS_FORECAST`` 可能预增也可能预亏、
``PRICE_MOVE`` 涨价对上游是好事对下游是坏事、``MONETARY`` 降准利好而加息利空。
这些必须靠词典在文本里读出方向，先验瞎给一个正负会系统性地错。
"""

_DOMAIN_KEYWORDS: Mapping[IntelDomain, tuple[str, ...]] = {
    IntelDomain.MACRO: (
        "央行",
        "货币政策",
        "CPI",
        "PPI",
        "PMI",
        "GDP",
        "社融",
        "汇率",
        "降准",
        "降息",
    ),
    IntelDomain.POLICY: (
        "证监会",
        "交易所",
        "国务院",
        "发改委",
        "监管",
        "新规",
        "办法",
        "征求意见",
    ),
    IntelDomain.COMPANY: ("公告", "股东", "董事会", "业绩", "减持", "增持", "回购", "立案"),
    IntelDomain.INDUSTRY: ("行业", "产业链", "景气", "供需", "出货量", "开工率", "库存"),
    IntelDomain.OVERSEAS: ("美联储", "美股", "港股", "纳斯达克", "地缘", "关税", "非农"),
    IntelDomain.MARKET: ("北向资金", "两融", "成交额", "涨停", "跌停", "龙虎榜", "融资余额"),
}

_POSITIVE_WORDS: tuple[str, ...] = (
    "增长",
    "回暖",
    "改善",
    "超预期",
    "创新高",
    "扩产",
    "中标",
    "获批",
    "签约",
    "利好",
    "上调",
    "提价",
    "涨价",
    "净买入",
    "扭亏",
    "盈利",
    "突破",
    "领先",
)
_NEGATIVE_WORDS: tuple[str, ...] = (
    "下滑",
    "回落",
    "恶化",
    "低于预期",
    "创新低",
    "停产",
    "亏损",
    "违规",
    "处罚",
    "利空",
    "下调",
    "降价",
    "净卖出",
    "爆雷",
    "逾期",
    "退市",
    "风险",
    "调查",
    "冻结",
)

_MAX_WORD_SENTIMENT = 0.6
"""词典打分的绝对值上限。词频堆出来的极端分没有意义，方向对了就够。"""


@dataclass(frozen=True, slots=True)
class ClassifyResult:
    """分类结果。"""

    event_type: EventType | None
    domain: IntelDomain
    classifier: str
    matched: tuple[str, ...]
    """命中的关键词，让分类可解释。"""


class EventClassifier:
    """基于关键词的事件与域分类器。"""

    def __init__(self, keywords: Mapping[EventType, Sequence[str]] | None = None) -> None:
        """初始化。

        Args:
            keywords: 事件关键词表，缺省用内置表。顺序即优先级。
        """
        self._keywords = {k: tuple(v) for k, v in (keywords or EVENT_KEYWORDS).items()}

    def classify(
        self, title: str, body: str = "", *, default_domain: IntelDomain | None = None
    ) -> ClassifyResult:
        """对一条情报分类。

        Args:
            title: 标题。
            body: 正文。
            default_domain: 源方已声明的域。声明了就不再猜。

        Returns:
            分类结果；无命中时 ``event_type`` 为 None，交由上层 LLM 兜底。
        """
        text = f"{title}\n{body}"
        for event, words in self._keywords.items():
            if hit := [w for w in words if w in text]:
                return ClassifyResult(
                    event_type=event,
                    domain=default_domain or self._infer_domain(text, event),
                    classifier="rule",
                    matched=tuple(hit),
                )
        return ClassifyResult(
            event_type=None,
            domain=default_domain or self._infer_domain(text, None),
            classifier="rule",
            matched=(),
        )

    @staticmethod
    def _infer_domain(text: str, event: EventType | None) -> IntelDomain:
        """推断情报域。

        Args:
            text: 全文。
            event: 已识别的事件类型。

        Returns:
            情报域。无线索时归入 MARKET。
        """
        if event is not None and (mapped := _EVENT_DOMAIN.get(event)) is not None:
            return mapped
        best: tuple[int, IntelDomain] = (0, IntelDomain.MARKET)
        for domain, words in _DOMAIN_KEYWORDS.items():
            score = sum(1 for w in words if w in text)
            if score > best[0]:
                best = (score, domain)
        return best[1]


_EVENT_DOMAIN: Mapping[EventType, IntelDomain] = {
    EventType.MONETARY: IntelDomain.MACRO,
    EventType.FISCAL: IntelDomain.MACRO,
    EventType.MACRO_DATA: IntelDomain.MACRO,
    EventType.GEOPOLITICS: IntelDomain.OVERSEAS,
    EventType.POLICY_POSITIVE: IntelDomain.POLICY,
    EventType.POLICY_NEGATIVE: IntelDomain.POLICY,
    EventType.PRICE_MOVE: IntelDomain.INDUSTRY,
    EventType.CAPACITY: IntelDomain.INDUSTRY,
    EventType.INDEX_ADJUST: IntelDomain.MARKET,
}


class SentimentScorer:
    """规则情绪打分。

    分数 = 事件先验 + 词典净得分，裁剪到 [-1, 1]。
    """

    def __init__(
        self,
        *,
        positive: Sequence[str] = _POSITIVE_WORDS,
        negative: Sequence[str] = _NEGATIVE_WORDS,
        priors: Mapping[EventType, float] | None = None,
    ) -> None:
        """初始化。

        Args:
            positive: 正面词。
            negative: 负面词。
            priors: 事件先验，缺省用内置表。
        """
        self._positive = tuple(positive)
        self._negative = tuple(negative)
        self._priors = dict(priors or EVENT_SENTIMENT_PRIOR)

    def score(self, title: str, body: str = "", *, event_type: EventType | None = None) -> float:
        """打分。

        标题权重是正文的两倍——财经快讯的判断信息几乎都在标题里，
        正文常含大段免责声明与背景介绍，等权会把信号稀释掉。

        Args:
            title: 标题。
            body: 正文。
            event_type: 事件类型，用于取先验。

        Returns:
            -1.0 ~ +1.0 的情绪分。
        """
        prior = self._priors.get(event_type, 0.0) if event_type is not None else 0.0
        lexical = 2 * self._lexical(title) + self._lexical(body)
        clipped = max(-_MAX_WORD_SENTIMENT, min(_MAX_WORD_SENTIMENT, lexical))
        return max(-1.0, min(1.0, prior + clipped))

    def _lexical(self, text: str) -> float:
        """词典净得分。

        Args:
            text: 文本。

        Returns:
            未裁剪的净得分。
        """
        if not text:
            return 0.0
        positive = sum(text.count(w) for w in self._positive)
        negative = sum(text.count(w) for w in self._negative)
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total * 0.3
