"""作者价格尺度与相对统计尺度独立冻结，不进行参数优化。"""
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache


@lru_cache(maxsize=8192)
def trading_days_since(as_of: date, on: date) -> int:
    """NYSE交易日数：不含as_of，含on；假期和周末不推进。"""
    if on <= as_of:
        return 0
    import exchange_calendars as xcals
    cal = xcals.get_calendar("XNYS", start=as_of - timedelta(days=7), end=on + timedelta(days=7))
    sessions = cal.sessions_in_range(as_of + timedelta(days=1), on)
    return len(sessions)


@dataclass(frozen=True)
class AuthorLevel:
    value: float
    as_of: date = date(2026, 9, 30)
    valid_sessions: int = 63
    unit: str = "USD"

    def expired(self, on: date) -> bool:
        return trading_days_since(self.as_of, on) > self.valid_sessions


AUTHOR_LEVELS = {
    "IXG": AuthorLevel(125),
    "MOVE": AuthorLevel(125, unit="index_points"),
    "AVGO": AuthorLevel(334),
    "SLV": AuthorLevel(53.3),
    "BWXT": AuthorLevel(131.8),
    "APP": AuthorLevel(277),
}


@dataclass(frozen=True)
class StatisticalConfig:
    relative_ma_days: int = 200
    move_window: int = 1260  # 最大5 × 252个收盘；最低252根才能计算分位。
    move_min_periods: int = 252
    move_quantile: float = 0.8
    stress_confirm_days: int = 2
    beta_days: int = 60
    yield_std_days: int = 10
    yield_median_days: int = 252
    yield_confirm_days: int = 5
    vix_window: int = 252
    vix_quantile: float = 0.8


STATISTICS = StatisticalConfig()
VERSION = "v1.1"
POLICY_REGIMES = ("HIKING", "PAUSE", "FIRST_CUT", "EASING")
YAHOO_SYMBOLS = ("IXG", "JNK", "QQQ", "SMH", "SPY", "SLV", "^VIX", "^TNX", "^MOVE", "TLT")
FRED_SYMBOLS = ("DGS10", "DGS2", "DFF", "BAMLH0A0HYM2", "PAYEMS")
DAILY_FRED = FRED_SYMBOLS[:-1]
