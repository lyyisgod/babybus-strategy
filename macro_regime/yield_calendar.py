"""v1.1：在各自公布日计算因子，仅将已计算结果向后映射至NYSE日期。"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import STATISTICS


@lru_cache(maxsize=64)
def nyse_sessions(start, end):
    import exchange_calendars as xcals
    cal = xcals.get_calendar("XNYS", start=start - pd.Timedelta(days=7),
                            end=end + pd.Timedelta(days=7))
    return cal.sessions_in_range(start, end)


def treasury_closed(day):
    """仅放行哥伦布日及退伍军人节全天休市；周日退伍军人节顺延周一。"""
    day = pd.Timestamp(day)
    columbus = pd.Timestamp(day.year, 10, 1)
    columbus += pd.Timedelta(days=(0 - columbus.weekday()) % 7 + 7)
    veterans = pd.Timestamp(day.year, 11, 11)
    if veterans.weekday() == 6:
        veterans += pd.Timedelta(days=1)
    return day in (columbus, veterans)


def publications(series, end):
    from .factors import DataError
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_unique \
            or not series.index.is_monotonic_increasing or series.index.tz is not None \
            or not series.index.equals(series.index.normalize()):
        raise DataError("公布日索引必须为无时区、递增且唯一的日期")
    observed = series.loc[:end].dropna().astype(float)
    if not np.isfinite(observed).all():
        raise DataError("公布序列含非有限值")
    return observed


def validate_dgs10(series, stock_dates):
    """检查真实NYSE缺口，不把正常休市补为收益率或零变化。"""
    from .factors import DataError
    observed = publications(series, stock_dates[-1])
    if observed.empty:
        raise DataError("缺少必需DGS10公布序列")
    sessions = nyse_sessions(min(stock_dates[0], observed.index[0]), stock_dates[-1])
    age = len(sessions[(sessions > observed.index[-1]) & (sessions <= stock_dates[-1])])
    if age > 5:
        raise DataError(f"DGS10公布过期: {age} NYSE交易日 > 5")
    active = sessions[sessions >= max(stock_dates[0], observed.index[0])]
    missing = active.difference(observed.index)
    faults = [d.date().isoformat() for d in missing if not treasury_closed(d)]
    if faults:
        raise DataError(f"DGS10缺少公布（非treasury_closed）: {faults}")
    return observed


def map_results(results, stock_dates, asof_name):
    """只带计算结果及publication_date；不带DGS10水平。"""
    # parquet与调用方可能分别使用us/ns精度；merge键统一精度，日期不变。
    left = pd.DataFrame({"stock_date": stock_dates.as_unit("ns")})
    right = results.rename_axis(asof_name).reset_index()
    right[asof_name] = right[asof_name].astype("datetime64[ns]")
    mapped = pd.merge_asof(left, right, left_on="stock_date", right_on=asof_name,
                           direction="backward")
    mapped = mapped.set_index("stock_date")
    mapped.index = stock_dates
    return mapped


def yield_factors(stock_dates, dgs10, move, cfg=STATISTICS):
    yields = validate_dgs10(dgs10, stock_dates)
    vol = yields.diff().rolling(cfg.yield_std_days, min_periods=cfg.yield_std_days).std(ddof=1)
    median = vol.rolling(cfg.yield_median_days, min_periods=200).median()
    from .factors import streak
    low = (vol < median).where(median.notna()).astype("boolean")
    yield_leg = (streak(low) >= cfg.yield_confirm_days).where(low.notna()).astype("boolean")
    yield_results = pd.DataFrame({"yield_std10": vol, "yield_std_median252": median,
                                  "yield_low_streak": streak(low), "yield_leg": yield_leg,
                                  "dgs10_delta5": yields.diff(5)})
    mapped_yield = map_results(yield_results, stock_dates, "yield_asof")
    observed_move = publications(move, stock_dates[-1])
    delta = observed_move.diff(5)
    declining = (delta <= 0).where(delta.notna()).astype("boolean")
    move_leg = (streak(declining) >= cfg.yield_confirm_days).where(declining.notna()).astype("boolean")
    move_results = pd.DataFrame({"move_delta5": delta, "move_nonrise_streak": streak(declining),
                                "move_yield_leg": move_leg})
    mapped_move = map_results(move_results, stock_dates, "move_asof")
    out = pd.concat([mapped_yield, mapped_move], axis=1)
    sessions = nyse_sessions(min(stock_dates[0], yields.index[0]), stock_dates[-1])
    for prefix, leg in (("yield", "yield_leg"), ("move", "move_yield_leg")):
        asof = out[f"{prefix}_asof"]
        ages = [np.nan if pd.isna(d) else int(sessions.searchsorted(t, side="right")
                 - sessions.searchsorted(d, side="right")) for t, d in zip(stock_dates, asof)]
        out[f"{prefix}_age_sessions"] = ages
        valid_asof = asof.notna()
        if prefix == "yield":
            valid_asof &= out[f"{prefix}_age_sessions"] <= 5
        out[leg] = out[leg].astype("boolean").where(valid_asof)
    # nullable逻辑严格：任一腿未知，最终就未知，即使另一腿为false。
    known = out.yield_leg.notna() & out.move_yield_leg.notna()
    out["yield_stable"] = (out.yield_leg & out.move_yield_leg).where(known).astype("boolean")
    out["dgs10_status"] = ["published" if t in yields.index else
                            "treasury_closed" if treasury_closed(t) else "before_history"
                            for t in stock_dates]
    return out
