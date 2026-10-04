"""Point-in-time historical frequencies for a fixed next-session close target.

This module pools eligible US stocks in the *same five-state bucket*.  It never
fits coefficients, accepts a supplied probability/label, or changes a trading
rule.  Frequencies and Wilson intervals are descriptive, not calibrated trading
probabilities; no out-of-sample advantage over a baseline has been established.

Both feature and adjusted-close records must have been verified and available
no later than their own XNYS session close.  Training labels must additionally
be known by the close preceding ``asof``.  Revisions published later are ignored.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
import math
from typing import Any

import exchange_calendars as xcals
import pandas as pd


BUCKET_FIELDS = ("regime", "stress", "dip_unit", "yield_stable", "at_252_high")
MIN_SAMPLE = 30
TARGET_RETURN = 0.05
TARGET_HORIZON_SESSIONS = 1
TARGET = {
    "calendar": "XNYS",
    "horizon_sessions": TARGET_HORIZON_SESSIONS,
    "price": "adj_close",
    "return_threshold": TARGET_RETURN,
    "comparison": ">=",
    "definition": "adj_close[next_XNYS_session] / adj_close[date] - 1 >= 0.05",
}
METHOD = "same_bucket_historical_frequency_wilson95"
EXCLUDED_SYMBOLS = frozenset({"GLD", "IAU", "SLV", "SILJ", "TLT", "GOLD", "SILVER"})
REGIMES = frozenset({"HIKING", "PAUSE", "FIRST_CUT", "EASING", "UNKNOWN"})
STRESS_STATES = frozenset({"NONE", "WATCH", "TRIGGER", "CONFIRMED_SELLOFF", "UNKNOWN"})
DIP_UNITS = frozenset({Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("1")})
_EQUITY_TYPES = frozenset({"EQUITY", "STOCK", "ETF"})
_Z95 = 1.959963984540054


def wilson(successes: int, n: int) -> tuple[float | None, float | None]:
    """Two-sided 95% Wilson score interval, without continuity correction."""
    if (isinstance(n, bool) or isinstance(successes, bool)
            or not isinstance(n, int) or not isinstance(successes, int)
            or n < 0 or successes < 0 or successes > n):
        raise ValueError("successes and n must be integers with 0 <= successes <= n")
    if not n:
        return None, None
    frequency = successes / n
    z2 = _Z95 * _Z95
    denominator = 1 + z2 / n
    center = (frequency + z2 / (2 * n)) / denominator
    radius = _Z95 * math.sqrt(frequency * (1 - frequency) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _day(value: Any) -> date | None:
    """Accept a session date, not an intraday timestamp or a guessed timezone."""
    try:
        if isinstance(value, str):
            return date.fromisoformat(value)
        if isinstance(value, datetime):
            return value.date() if value.time().replace(tzinfo=None) == datetime.min.time() else None
        if isinstance(value, date):
            return value
    except (TypeError, ValueError):
        pass
    return None


def _available(value: Any) -> pd.Timestamp | None:
    # Numeric values are deliberately rejected: a bare epoch has no declared
    # timezone or unit, and pandas otherwise guesses nanoseconds.
    if not isinstance(value, (str, datetime, pd.Timestamp)):
        return None
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            return None
        return stamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        return None


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _bucket(value: Any) -> tuple[Any, ...] | None:
    if not isinstance(value, Mapping):
        return None
    regime, stress = value.get("regime"), value.get("stress")
    unit = _decimal(value.get("dip_unit"))
    stable, high = value.get("yield_stable"), value.get("at_252_high")
    if (not isinstance(regime, str) or regime not in REGIMES
            or not isinstance(stress, str) or stress not in STRESS_STATES or unit not in DIP_UNITS
            or not isinstance(stable, bool) or not isinstance(high, bool)):
        return None
    return regime, stress, unit, stable, high


def _symbol(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _excluded(symbol: str, market: Any = "US", asset_type: Any = "EQUITY") -> bool:
    return (symbol in EXCLUDED_SYMBOLS or not isinstance(market, str)
            or market.upper() != "US" or not isinstance(asset_type, str)
            or asset_type.upper() not in _EQUITY_TYPES)


@lru_cache(maxsize=16)
def _calendar(start: date, end: date):
    return xcals.get_calendar("XNYS", start=start, end=end)


def _valid_time(record: Mapping, on: date, cal, cutoff: pd.Timestamp) -> pd.Timestamp | None:
    if (record.get("verified") is not True or not isinstance(record.get("source"), str)
            or not record["source"].strip()):
        return None
    available = _available(record.get("available_at"))
    if (available is None or available > cutoff or on < cal.first_session.date()
            or on > cal.last_session.date() or not cal.is_session(on.isoformat())
            or available > cal.session_close(on.isoformat())):
        return None
    return available


def _put_latest(records: dict, key: Any, available: pd.Timestamp, value: Any) -> None:
    """Deduplicate versions; conflicting values at the latest timestamp abstain."""
    previous = records.get(key)
    if previous is None or available > previous[0]:
        records[key] = (available, value)
    elif available == previous[0] and value != previous[1]:
        records[key] = (available, None)


def _result(symbol: str, asof: Any, reason: str, n: int = 0) -> dict:
    on = _day(asof)
    return {
        "symbol": symbol,
        "asof": on.isoformat() if on else str(asof),
        "target": dict(TARGET),
        "p": None,
        "n": n,
        "ci_low": None,
        "ci_high": None,
        "method": METHOD,
        "calibrated": False,
        "beats_baseline": None,
        "abstain": True,
        "reason": reason,
    }


def estimate(symbol, asof, bucket, *, panel=None, market="US", asset_type="EQUITY") -> dict:
    """Estimate the pooled same-bucket frequency, using only matured PIT labels.

    ``panel`` must have ``pit_verified=True``, ``rows`` and ``prices``.  Rows
    supply historical five-state features; prices map each symbol to its raw
    adjusted-close records.  Every source, timestamp, verified/closed flag and
    next XNYS session is checked independently.  Supplied labels, probabilities
    or intraday highs are ignored.  Missing next-session prices are never bridged.

    Invalid/incomplete records are excluded, rather than filled.  With fewer
    than 30 eligible symbol/session observations, probability and interval are
    null.  Otherwise ``abstain=False`` means an empirical frequency is available;
    ``calibrated=False`` and ``beats_baseline=None`` remain explicit.
    """
    symbol = _symbol(symbol)
    if _excluded(symbol, market, asset_type):
        return _result(symbol, asof, "excluded_asset")
    if (not isinstance(panel, Mapping) or panel.get("pit_verified") is not True
            or not isinstance(panel.get("rows"), (list, tuple))
            or not isinstance(panel.get("prices"), Mapping)):
        return _result(symbol, asof, "no_pit_panel")
    on = _day(asof)
    if on is None:
        return _result(symbol, asof, "invalid_asof")
    wanted = _bucket(bucket)
    if wanted is None:
        return _result(symbol, asof, "invalid_bucket")

    # Build the exchange calendar locally.  No research case, network service,
    # frozen macro_regime module or user-supplied label is consulted.
    try:
        cal = _calendar(on - timedelta(days=17), on + timedelta(days=14))
        if not cal.is_session(on.isoformat()):
            return _result(symbol, asof, "invalid_asof")
        previous = cal.previous_session(on.isoformat())
        cutoff = cal.session_close(previous)
        cutoff_day = previous.date()
        # Discard unavailable revisions *before* choosing the historical
        # calendar range, so a future backfill cannot even alter that range.
        dates = [on - timedelta(days=10)]
        for record in panel["rows"]:
            if not isinstance(record, Mapping):
                continue
            day = _day(record.get("date"))
            available = _available(record.get("available_at"))
            if (day and day <= cutoff_day and available is not None and available <= cutoff
                    and record.get("verified") is True
                    and not _excluded(_symbol(record.get("symbol")), record.get("market", "US"),
                                      record.get("asset_type", "EQUITY"))):
                dates.append(day)
        cal = _calendar(min(dates) - timedelta(days=7), on + timedelta(days=14))
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
        return _result(symbol, asof, "invalid_asof")

    historical: dict[tuple[str, date], tuple[pd.Timestamp, tuple | None]] = {}
    for row in panel["rows"]:
        if not isinstance(row, Mapping):
            continue
        historical_symbol = _symbol(row.get("symbol"))
        if not historical_symbol or _excluded(historical_symbol, row.get("market", "US"),
                                               row.get("asset_type", "EQUITY")):
            continue
        day = _day(row.get("date"))
        if day is None or day > cutoff_day:
            continue
        available = _valid_time(row, day, cal, cutoff)
        state = _bucket(row)
        if available is not None and state is not None:
            _put_latest(historical, (historical_symbol, day), available, state)

    prices: dict[str, dict[date, tuple[pd.Timestamp, Decimal | None]]] = {}
    relevant_symbols = {key[0] for key in historical}
    for raw_symbol, series in panel["prices"].items():
        price_symbol = _symbol(raw_symbol)
        if price_symbol not in relevant_symbols or not isinstance(series, (list, tuple)):
            continue
        selected = prices.setdefault(price_symbol, {})
        for record in series:
            if not isinstance(record, Mapping) or record.get("closed") is not True:
                continue
            day = _day(record.get("date"))
            if day is None or day > cutoff_day:
                continue
            # An orphan price older than the calendar has no eligible feature
            # row; it cannot be either leg of any target in this panel.
            if day < cal.first_session.date():
                continue
            available = _valid_time(record, day, cal, cutoff)
            close = _decimal(record.get("adj_close"))
            if available is not None and close is not None and close > 0:
                _put_latest(selected, day, available, close)

    successes = n = 0
    for (historical_symbol, day), (_, state) in historical.items():
        if state != wanted:
            continue
        next_day = cal.next_session(day.isoformat()).date()
        if next_day > cutoff_day:
            continue
        series = prices.get(historical_symbol, {})
        current, following = series.get(day), series.get(next_day)
        if not current or not following or current[1] is None or following[1] is None:
            continue
        n += 1
        # Comparing the exact 1.05 multiple avoids division rounding at the
        # inclusive threshold, including high-precision decimal input strings.
        with localcontext() as context:
            context.prec = max(28, len(current[1].as_tuple().digits) + 3)
            successes += int(following[1] >= current[1] * Decimal("1.05"))

    result = _result(symbol, asof, "insufficient_sample", n)
    if n >= MIN_SAMPLE:
        low, high = wilson(successes, n)
        result.update(p=successes / n, ci_low=low, ci_high=high, abstain=False,
                      reason="historical_frequency_uncalibrated")
    return result
