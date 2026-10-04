"""Classify a policy cycle from verified official statements only.

This is a deterministic evidence contract, not an inference from market prices
or futures probabilities. Dates are calendar dates; the caller must supply the
complete available statement sequence, including unchanged-rate meetings.
"""
from collections.abc import Mapping
from datetime import date, datetime
import math
from numbers import Real
from urllib.parse import urlparse

from .inputs import nyse


def _calendar_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        parsed = date.fromisoformat(value)
        if parsed.isoformat() == value:
            return parsed
    raise ValueError("date must be YYYY-MM-DD")


def _validated_statement(record):
    source = record.get("source")
    if not source:
        raise ValueError("missing_official_source")
    if not isinstance(source, str):
        raise ValueError("nonofficial_statement")
    try:
        parsed = urlparse(source)
        official = parsed.scheme == "https" and parsed.hostname in {
            "federalreserve.gov", "www.federalreserve.gov"
        }
    except ValueError:
        official = False
    if not official:
        raise ValueError("nonofficial_statement")
    if record.get("verified") is not True:
        raise ValueError("unverified_statement")
    action = record.get("action")
    if not isinstance(action, str) or action not in {"HIKE", "HOLD", "CUT"}:
        raise ValueError("invalid_statement_action")
    target = record.get("target_range")
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        raise ValueError("invalid_target_range")
    if not all(isinstance(v, Real) and not isinstance(v, bool)
               and math.isfinite(v) for v in target):
        raise ValueError("invalid_target_range")
    low, high = target
    if low >= high:
        raise ValueError("invalid_target_range")
    return action, (low, high)


def official_policy(statements, asof):
    """Return ``regime, cut_index, reason, last_statement_date``.

    Each record requires ``date``, ``action`` (HIKE/HOLD/CUT),
    ``target_range`` (two percentage values), an official HTTPS ``source``,
    and ``verified is True``. Future records are excluded before validating
    their remaining fields. Optional ``available_at`` must be timezone-aware;
    records unavailable at the requested XNYS close (or the previous session
    close on a holiday) are also excluded before validating other fields. If
    absent, the caller attests that the official statement was known on its
    declared publication date. Identical same-date decisions are deduplicated;
    contradictory same-date decisions invalidate the supplied history.

    A HIKE establishes or restarts the cycle. Its subsequent first CUT is
    FIRST_CUT, further cuts are EASING, and a post-cut HOLD remains EASING.
    Without a HIKE anchor, an exact cut index cannot be proved. Known hiking
    or pause states use index 0; UNKNOWN uses None. A first-record HIKE is an
    allowed anchor, with an explicit trace that no earlier range was supplied.
    """
    result = {"regime": "UNKNOWN", "cut_index": None,
              "reason": [], "last_statement_date": None}

    def unknown(reason):
        return {**result, "reason": [reason]}

    try:
        cutoff = _calendar_date(asof)
    except (TypeError, ValueError):
        return unknown("invalid_asof")
    if statements is None:
        return unknown("missing_official_statements")
    try:
        records = list(statements)
    except TypeError:
        return unknown("invalid_statements")

    calendar = nyse(cutoff.year)
    session = calendar.date_to_session(cutoff.isoformat(), direction="previous")
    close = calendar.session_close(session).to_pydatetime()

    eligible = []
    date_error = None
    for record in records:
        if not isinstance(record, Mapping):
            date_error = "invalid_statement_date"
            continue
        if "available_at" in record:
            try:
                value = record["available_at"]
                available = value if isinstance(value, datetime) else datetime.fromisoformat(value)
                if available.tzinfo is None or available.utcoffset() is None:
                    raise ValueError("available_at requires a timezone")
            except (TypeError, ValueError):
                date_error = "invalid_statement_available_at"
                continue
            if available > close:
                continue
        try:
            observed = _calendar_date(record.get("date"))
        except (TypeError, ValueError):
            date_error = "invalid_statement_date"
            continue
        if observed <= cutoff:
            eligible.append((observed, record))
    eligible.sort(key=lambda item: item[0])
    if eligible:
        result["last_statement_date"] = eligible[-1][0].isoformat()
    if date_error:
        return unknown(date_error)
    if not eligible:
        return unknown("no_statements_on_or_before_asof")

    history = []
    for observed, record in eligible:
        try:
            action, target = _validated_statement(record)
        except (TypeError, ValueError) as error:
            return unknown(str(error))
        if history and observed == history[-1][0]:
            if (action, target) != history[-1][1:]:
                return unknown("contradictory_statements")
            continue
        if history:
            old_target = history[-1][2]
            differences = tuple(new - old for new, old in zip(target, old_target))
            consistent = (
                action == "HIKE" and all(change > 0 for change in differences)
                or action == "CUT" and all(change < 0 for change in differences)
                or action == "HOLD" and all(change == 0 for change in differences)
            )
            if not consistent:
                return unknown("contradictory_statements")
        history.append((observed, action, target))

    anchored = False
    cut_index = 0
    regime = "UNKNOWN"
    for _, action, _ in history:
        if action == "HIKE":
            anchored = True
            cut_index = 0
            regime = "HIKING"
        elif not anchored:
            continue
        elif action == "CUT":
            cut_index += 1
            regime = "FIRST_CUT" if cut_index == 1 else "EASING"
        else:
            regime = "EASING" if cut_index else "PAUSE"
    if not anchored:
        return unknown("missing_hike_anchor")
    trace = (["initial_hike_anchor_without_prior_range"]
             if history[0][1] == "HIKE" else [])
    return {**result, "regime": regime, "cut_index": cut_index, "reason": trace}
