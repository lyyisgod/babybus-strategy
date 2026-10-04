"""Point-in-time macro observations; policy and portfolio decisions live elsewhere.

The frozen v1.1 pressure combinations and hysteresis are reused unchanged.
Publication factors are calculated on publication dates after a one-publication
lag, then mapped backward to the observed NYSE session.  No caller-supplied
returns, pressure states, or differentials are consumed.
"""
from __future__ import annotations

from datetime import date, datetime
from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from macro_regime.config import AUTHOR_LEVELS, STATISTICS
from macro_regime.factors import DataError, dip_units, rsi, streak
from macro_regime.state import _confirm_pressure, pressure_combinations
from macro_regime.yield_calendar import (
    map_results, nyse_sessions, treasury_closed, validate_dgs10,
)

_PRICES = ("IXG", "SPY", "JNK", "QQQ", "SMH", "TLT")
_PUBLICATIONS = ("DGS10", "OAS", "MOVE")
_TZ = "America/New_York"


def _date(value):
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return pd.Timestamp(value).tz_convert(_TZ).normalize().tz_localize(None)
        return pd.Timestamp(value.date())
    if isinstance(value, date):
        return pd.Timestamp(value)
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
        raise DataError("date must be a timezone-free calendar date")
    return parsed


def _number(value, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise DataError("boolean is not a numeric observation")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise DataError("observation must be finite" + (" and positive" if positive else ""))
    return number


def _json(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value) if isinstance(value, (float, np.floating)) else value


def _empty(asof, reasons):
    return {
        "stress": {"state": "UNKNOWN", "author_combo": False, "stat_combo": False,
                   "pair": None, "pair_prev": None, "trigger_on": False,
                   "pair_streak": 0, "clear_streak": 0},
        "pressure": False, "yield_stable": None, "data_valid": False,
        "trace": {"asof": str(asof), "reasons": reasons},
    }


def _calendar(start, end):
    import exchange_calendars as xcals
    return xcals.get_calendar("XNYS", start=start - pd.Timedelta(days=7),
                              end=end + pd.Timedelta(days=7))


def _read_rows(rows, name, requested, cutoff, calendar, *, prices, rejected):
    """Select the most recent eligible vintage, never a later price revision."""
    eligible = {}
    if not isinstance(rows, (list, tuple)):
        rejected.append(f"{name}: missing raw rows")
        return pd.DataFrame()
    for row in rows:
        try:
            if not isinstance(row, Mapping):
                raise DataError("row must be an object")
            on = _date(row.get("date"))
            # Future records do not enter either factors or diagnostics.
            if on > requested:
                continue
            available = pd.Timestamp(row.get("available_at"))
            if pd.isna(available) or available.tzinfo is None:
                raise DataError("available_at requires a timezone")
            available = available.tz_convert("UTC")
            if available > cutoff:
                continue
            if row.get("verified") is not True or not isinstance(row.get("source"), str) \
                    or not row["source"].strip():
                raise DataError("verified source required")
            if available.tz_convert(_TZ).date() < on.date():
                raise DataError("available_at precedes observation date")
            if prices:
                if row.get("closed") is not True:
                    raise DataError("completed close required")
                if not calendar.is_session(on):
                    raise DataError("price date is not an XNYS session")
                # An as-of snapshot cannot recreate a historical signal from a
                # revision that first became available after its own close.
                if available > calendar.session_close(on):
                    continue
                values = {"close": _number(row.get("close"), positive=True),
                          "adj_close": _number(row.get("adj_close"), positive=True)}
            else:
                values = {"value": _number(row.get("value"), positive=name == "MOVE")}
            if on not in eligible or available > eligible[on]["available_at"]:
                eligible[on] = {**values, "available_at": available, "source": row["source"]}
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            rejected.append(f"{name}: {exc}")
    if not eligible:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(eligible, orient="index").sort_index()


def _publication_map(series, sessions, name):
    # Both operations take place before mapping to the stock-date coordinate.
    delta = series.shift(1).diff(5)
    return map_results(pd.DataFrame({name: delta}), sessions, name + "_asof")


def _yield_factors(sessions, dgs10, move):
    """v1.1 yield algorithm, with the required publication-coordinate lag.

    Validate original DGS10 publications before shifting: the first lagged NaN
    is warm-up, and a Treasury closure is never a synthetic zero publication.
    """
    cfg = STATISTICS
    observed = validate_dgs10(dgs10, sessions)
    yields = observed.shift(1)
    vol = yields.diff().rolling(cfg.yield_std_days, min_periods=cfg.yield_std_days).std(ddof=1)
    median = vol.rolling(cfg.yield_median_days, min_periods=200).median()
    low = (vol < median).where(median.notna()).astype("boolean")
    yield_leg = (streak(low) >= cfg.yield_confirm_days).where(low.notna()).astype("boolean")
    yield_results = pd.DataFrame({
        "yield_std10": vol, "yield_std_median252": median,
        "yield_low_streak": streak(low), "yield_leg": yield_leg,
        "dgs10_delta5": yields.diff(5),
    })
    mapped_yield = map_results(yield_results, sessions, "yield_asof")
    delta = move.shift(1).diff(5)
    declining = (delta <= 0).where(delta.notna()).astype("boolean")
    move_leg = (streak(declining) >= cfg.yield_confirm_days).where(declining.notna()).astype("boolean")
    move_results = pd.DataFrame({"move_delta5": delta, "move_nonrise_streak": streak(declining),
                                 "move_yield_leg": move_leg})
    mapped_move = map_results(move_results, sessions, "move_asof")
    out = pd.concat([mapped_yield, mapped_move], axis=1)
    clock = nyse_sessions(min(sessions[0], observed.index[0]), sessions[-1])
    for prefix, leg in (("yield", "yield_leg"), ("move", "move_yield_leg")):
        dates = out[f"{prefix}_asof"]
        out[f"{prefix}_age_sessions"] = [
            np.nan if pd.isna(d) else int(clock.searchsorted(t, side="right")
                                          - clock.searchsorted(d, side="right"))
            for t, d in zip(sessions, dates)
        ]
        valid = dates.notna()
        if prefix == "yield":
            valid &= out[f"{prefix}_age_sessions"] <= 5
        out[leg] = out[leg].astype("boolean").where(valid)
    known = out.yield_leg.notna() & out.move_yield_leg.notna()
    out["yield_stable"] = (out.yield_leg & out.move_yield_leg).where(known).astype("boolean")
    out["dgs10_status"] = ["published" if t in observed.index else
                            "treasury_closed" if treasury_closed(t) else "before_history"
                            for t in sessions]
    return out


def _move_series(publications, prices, sessions, calendar):
    """Use one identity throughout: observed MOVE, or the complete TLT proxy."""
    actual = pd.Series(dtype=float)
    if not publications.empty:
        # The raw index used for pressure must be known at that session close.
        actual = publications.loc[
            [calendar.is_session(d) and row.available_at <= calendar.session_close(d)
             for d, row in publications.iterrows()], "value"
        ].astype(float)
    aligned = actual.reindex(sessions)
    first = aligned.first_valid_index()
    gap = first is not None and aligned.loc[first:].isna().any()
    use_proxy = first is None or pd.isna(aligned.iloc[-1]) or bool(gap)
    reason = "actual_MOVE_history_gap" if gap else "actual_MOVE_unavailable_at_close" if use_proxy else None
    if not use_proxy:
        return aligned, publications.value.astype(float), False, reason
    tlt = prices["TLT"].get("adj_close", pd.Series(dtype=float)).reindex(sessions)
    proxy = np.log(tlt).diff().rolling(20, min_periods=20).std(ddof=1) * math.sqrt(252) * 100
    # Derivation dates are the proxy's publication coordinate. Warm-up NaNs
    # are excluded rather than inserted as fake observations.
    return proxy, proxy.dropna(), True, reason


def macro_facts(facts, asof) -> dict:
    """Calculate observed macro facts from verified point-in-time raw rows.

    Missing ancillary publications invalidate additions without erasing an
    independently confirmed pressure combination. An unprovable pressure
    state is UNKNOWN, and insufficient yield warm-up remains nullable.
    """
    try:
        requested = _date(asof)
    except (ValueError, TypeError, OverflowError) as exc:
        return _empty(asof, [f"invalid_asof: {exc}"])
    if not isinstance(facts, Mapping):
        return _empty(asof, ["facts must be an object"])
    raw_prices = facts.get("macro_prices", {})
    raw_publications = facts.get("publications", {})
    if not isinstance(raw_prices, Mapping) or not isinstance(raw_publications, Mapping):
        return _empty(asof, ["macro_prices and publications must be objects"])
    # Establish the observation cutoff without consulting any input rows.
    # Unavailable backfills must not change even the calendar's construction
    # range, including when they carry a date preceding all visible history.
    observation_calendar = _calendar(requested, requested)
    session = observation_calendar.date_to_session(requested, direction="previous")
    cutoff = observation_calendar.session_close(session)
    starts = [requested]
    for group in (raw_prices, raw_publications):
        for name in (*_PRICES, *_PUBLICATIONS):
            rows = group.get(name, [])
            if isinstance(rows, (list, tuple)):
                for row in rows:
                    try:
                        on = _date(row.get("date"))
                        available = pd.Timestamp(row.get("available_at"))
                        if on <= requested and pd.notna(available) and available.tzinfo is not None \
                                and available.tz_convert("UTC") <= cutoff:
                            starts.append(on)
                    except (AttributeError, ValueError, TypeError, OverflowError):
                        pass
    calendar = _calendar(min(starts), requested)
    # A holiday request reports the most recent completed observation. The
    # caller owns scheduling; the cutoff remains that completed session close.
    rejected = []
    prices = {name: _read_rows(raw_prices.get(name), name, requested, cutoff, calendar,
                               prices=True, rejected=rejected) for name in _PRICES}
    publications = {name: _read_rows(raw_publications.get(name), name, requested, cutoff, calendar,
                                     prices=False, rejected=rejected) for name in _PUBLICATIONS}
    usable = [p.index[0] for p in prices.values() if not p.empty]
    if not usable:
        result = _empty(asof, ["no eligible completed macro prices"])
        result["trace"].update({"observed_session": session.date().isoformat(),
                                "cutoff": cutoff.isoformat(), "rejected_rows": rejected})
        return result
    sessions = calendar.sessions_in_range(min(usable), session).tz_localize(None)
    if sessions.empty:
        return _empty(asof, ["no eligible XNYS observation sessions"])
    reasons = []
    valid = True
    frame = pd.DataFrame(index=sessions)
    for name in _PRICES:
        for field, suffix in (("close", ""), ("adj_close", "_adj")):
            frame[name + suffix] = prices[name].get(field, pd.Series(dtype=float)).reindex(sessions)
    move, yield_move, proxy, proxy_reason = _move_series(publications["MOVE"], prices, sessions, calendar)
    frame["MOVE"] = move
    relative = frame.IXG_adj / frame.SPY_adj
    ma = relative.rolling(STATISTICS.relative_ma_days).mean()
    q80 = move.rolling(STATISTICS.move_window, min_periods=STATISTICS.move_min_periods).quantile(
        STATISTICS.move_quantile)
    percentile = move.rolling(STATISTICS.move_window, min_periods=STATISTICS.move_min_periods).apply(
        lambda a: np.nan if not np.isfinite(a[-1]) else
        100 * ((a < a[-1]).sum() + .5 * (a == a[-1]).sum()) / np.isfinite(a).sum(), raw=True)
    counts = move.rolling(STATISTICS.move_window, min_periods=0).count()
    combinations = []
    ixg_level, move_level = AUTHOR_LEVELS["IXG"], AUTHOR_LEVELS["MOVE"]
    for d in sessions:
        inactive = d.date() < ixg_level.as_of or d.date() < move_level.as_of \
            or ixg_level.expired(d.date()) or move_level.expired(d.date())
        values_known = bool(np.isfinite([frame.loc[d, "IXG"], move.loc[d], relative.loc[d]]).all())
        statistical_known = bool(np.isfinite([ma.loc[d], percentile.loc[d]]).all())
        author_known = not inactive and not proxy
        combos = pressure_combinations(inactive, proxy, frame.loc[d, "IXG"] < ixg_level.value,
                                       move.loc[d] > move_level.value,
                                       relative.loc[d] < ma.loc[d], _json(percentile.loc[d]))
        combinations.append({**combos, "known": values_known and (author_known or statistical_known)})
    raw_pressure = pd.DataFrame(combinations, index=sessions)
    history = _confirm_pressure(raw_pressure, STATISTICS.stress_confirm_days)
    last = history.iloc[-1]
    stress_known = len(sessions) >= STATISTICS.stress_confirm_days \
        and bool(raw_pressure.known.tail(STATISTICS.stress_confirm_days).all())
    stress = {k: _json(last[k]) for k in last.index if k != "state"}
    stress.update({"state": last.state if stress_known else "UNKNOWN", "move_pct": _json(percentile.iloc[-1]),
                   "move_observations": int(counts.iloc[-1]), "move_proxy": proxy})
    pressure = stress["state"] == "TRIGGER"
    if not stress_known:
        valid = False
        reasons.append("stress_confirmation_unavailable: need two known complete pressure observations")
    if proxy_reason:
        reasons.append(proxy_reason + "; entire history uses TLT proxy")
    if proxy and pd.isna(percentile.iloc[-1]):
        reasons.append("MOVE_proxy_warmup: at least 252 actual proxy observations required")
    elif pd.isna(percentile.iloc[-1]):
        reasons.append("MOVE_percentile_unavailable; author combination may remain valid")
    returns = {}
    for name in ("JNK", "QQQ", "SMH"):
        close = frame[name + "_adj"]
        returns[name] = {"r1": _json(close.pct_change(fill_method=None).iloc[-1]),
                         "r5": _json(close.pct_change(5, fill_method=None).iloc[-1]),
                         "rsi14": _json(rsi(close).iloc[-1]),
                         "new_low20": bool((close < close.shift().rolling(20).min()).iloc[-1])}
    jnk_streak = int(streak(frame.JNK_adj.pct_change(fill_method=None) < 0).iloc[-1])
    for name in ("JNK", "QQQ", "SMH"):
        if returns[name]["r1"] is None or returns[name]["r5"] is None \
                or frame[name + "_adj"].tail(6).isna().any():
            valid = False
            reasons.append(f"{name}_return_history_unavailable")
    deltas = {"oas_delta5": None, "dgs10_delta5": None, "move_delta5": None}
    yield_calendar = {}
    stable = None
    for name in ("DGS10", "OAS"):
        if publications[name].empty:
            valid = False
            reasons.append(f"{name}_publications_unavailable")
    if not publications["OAS"].empty:
        oas = _publication_map(publications["OAS"].value.astype(float), sessions, "oas_delta5").iloc[-1]
        deltas["oas_delta5"] = _json(oas.oas_delta5)
        yield_calendar["oas_asof"] = _json(oas.oas_delta5_asof)
        if deltas["oas_delta5"] is None:
            valid = False
            reasons.append("OAS_publication_difference_warmup")
    if not publications["DGS10"].empty:
        try:
            y = _yield_factors(sessions, publications["DGS10"].value.astype(float), yield_move).iloc[-1]
            stable = _json(y.yield_stable)
            for name in ("dgs10_delta5", "move_delta5"):
                deltas[name] = _json(y[name])
            yield_calendar.update({name: _json(y[name]) for name in y.index if name not in deltas})
            if deltas["dgs10_delta5"] is None:
                valid = False
                reasons.append("DGS10_publication_difference_warmup")
            if stable is None:
                reasons.append("yield_stable_warmup_or_unknown_leg")
        except (DataError, ValueError, TypeError) as exc:
            valid = False
            reasons.append(f"DGS10_publication_validation: {exc}")
    # The confirmed selloff classification needs just these two price returns
    # and confirmed pressure, irrespective of rate/credit availability.
    j, q, s = (returns[name]["r5"] for name in ("JNK", "QQQ", "SMH"))
    o, d = deltas["oas_delta5"], deltas["dgs10_delta5"]
    if j is not None and q is not None and j < -.015 and q < -.015 and pressure:
        divergence = "CONFIRMED_SELLOFF"
    elif j is not None and o is not None and d is not None and j < -.015 and o <= 0 and d > 0:
        divergence = "RATE_NOT_CREDIT"
    elif j is not None and s is not None and (j < -.015 or jnk_streak >= 5) and s >= -.01:
        divergence = "UNCONFIRMED_CREDIT"
    elif j is not None and q is not None and j > 0 and q < 0:
        divergence = "RISK_ON_DIVERGENCE"
    elif j is None or q is None or s is None:
        divergence = "UNKNOWN"
    else:
        divergence = "NONE"
    price_trace = {"IXG_close": _json(frame.IXG.iloc[-1]), "relative_ixg": _json(relative.iloc[-1]),
                   "relative_ma200": _json(ma.iloc[-1]), "JNK_down_streak": jnk_streak,
                   "SMH_dip_units": None}
    if returns["SMH"]["r1"] is not None and s is not None:
        price_trace["SMH_dip_units"] = dip_units(returns["SMH"]["r1"], s)
    trace = {
        "asof": requested.date().isoformat(), "observed_session": session.date().isoformat(),
        "asof_is_session": bool(calendar.is_session(requested)), "cutoff": cutoff.isoformat(),
        "divergence": divergence, "returns": returns, "publication_deltas": deltas,
        "yield_calendar": yield_calendar, "price_indicators": price_trace,
        "move": {"proxy": proxy, "identity": "TLT_adjusted_log_return_20d_sample_std_annualized_pct"
                 if proxy else "MOVE", "value": _json(move.iloc[-1]), "q80": _json(q80.iloc[-1]),
                 "percentile": _json(percentile.iloc[-1]), "observations": int(counts.iloc[-1])},
        "publication_units": "original_percentage_points_for_DGS10_and_OAS",
        "publication_coordinate": "shift(1) then diff(5), then backward asof",
        "reasons": reasons, "rejected_rows": rejected,
        "sources": {name: sorted(set(table.source)) if not table.empty else []
                    for name, table in {**prices, **publications}.items()},
    }
    return {"stress": stress, "pressure": bool(pressure), "yield_stable": stable,
            "trace": trace, "data_valid": bool(valid)}
