"""只组合已冻结的数值引擎；官方政策/PENDING/OAS 发布滞后作为接口契约。

不调用 legacy decide，也不把 UNKNOWN 伪装成 HIKING；不产生股票名单。
"""
import math
import numpy as np
import pandas as pd

from macro_regime.config import VERSION
from macro_regime.factors import DataError, numeric_factors, author_levels, dip_units
from macro_regime.state import pressure_history, divergence


def lagged_oas(publications, dates):
    """输入必须是实际公布日期索引，不是未经核验的观测日期。

    在第 n 次公布只能使用第 n-1 次数值。先在公布坐标计算变化，再向后
    映射派生结果；未来行在任何 rolling/shift 之前截断。
    """
    s = publications.loc[:dates[-1]].dropna()
    if not isinstance(s.index, pd.DatetimeIndex) or not s.index.is_unique or not s.index.is_monotonic_increasing:
        raise DataError("OAS actual publication dates must be sorted and unique")
    if s.empty or not np.isfinite(s).all():
        raise DataError("missing OAS publications")
    lag = s.shift(1)
    right = pd.DataFrame({"date": pd.DatetimeIndex(s.index).as_unit("ns"),
                          "oas_level": lag.values, "oas_delta5": lag.diff(5).values,
                          "oas_used_publication": pd.Series(s.index, index=s.index).shift(1).values})
    result = pd.merge_asof(pd.DataFrame({"date": pd.DatetimeIndex(dates).as_unit("ns")}),
                           right, on="date", direction="backward")
    result.index = dates
    return result.drop(columns="date")


def instrument_rules(returns):
    """由冻结宏观 sizing 函数输出序数，选择层不重新解释跌幅分档。"""
    return {symbol: {"size_unit": dip_units(r["ret_1d"], r["ret_5d"]),
                     "chase_block": r["ret_5d"] > .08,
                     "asof": r["asof"]} for symbol, r in returns.items()}


def macro_output(dataset, policy, *, oas_publications, oas_metadata, asset_returns=None):
    if not oas_metadata.get("publication_dates_verified"):
        raise DataError("OAS publication schedule unverified; observation dates are not release dates")
    frame = dataset.frame
    factors = numeric_factors(frame, dgs10_publications=dataset.dgs10_publications,
                              move_publications=dataset.move_publications)
    oas = lagged_oas(oas_publications, frame.index)
    factors["oas_delta5"] = oas.oas_delta5
    pressure = pressure_history(frame, factors)
    row, p = factors.iloc[-1], pressure.iloc[-1]
    required = ["JNK_r5", "QQQ_r5", "SMH_r5", "oas_delta5", "dgs10_delta5"]
    if any(pd.isna(row[k]) for k in required) or pd.isna(p.state):
        raise DataError("missing required macro factor history")
    stress = p.state
    if stress != "TRIGGER" and bool(p.pair):
        stress = "PENDING"  # 只重命名首次完整组合；两次确认/解除仍由 v1.1 计算。
    div = divergence(row, stress)
    pressure_on = stress == "TRIGGER" or div == "CONFIRMED_SELLOFF"
    stable = None if pd.isna(row.yield_stable) else bool(row.yield_stable)
    dff = frame.DFF.dropna()
    # Dataset 可含非 NYSE 公布日；调用方可提供原频率 DFF 供标签使用。
    native_dff = getattr(dataset, "dff_publications", None)
    if native_dff is not None:
        dff = native_dff.loc[:frame.index[-1]].dropna()
    hike_proxy = None if native_dff is None or len(dff) < 64 else bool(dff.iloc[-1] > dff.iloc[-64])
    return {"version": VERSION, "asof": frame.index[-1].date().isoformat(),
            "data_valid": True, "sources": dataset.metadata["series"], **policy,
            "stress": {"state": stress, **{k: bool(p[k]) for k in
                       ("author_combo", "stat_combo", "pair", "pair_prev", "trigger_on")},
                       "pair_streak": int(p.pair_streak), "clear_streak": int(p.clear_streak),
                       "move_pct": None if pd.isna(row.move_pct5y) else float(row.move_pct5y)},
            "divergence": div, "yield_stable": stable,
            "balance_sheet_stress": bool(pressure_on and row.dgs10_delta5 > 0 and row.oas_delta5 > 0),
            "constraints": {"allow_margin": not pressure_on, "max_gross": 1.0 if pressure_on else None},
            "hike_proxy": hike_proxy, "author_levels": author_levels(frame),
            "asset_rules": instrument_rules(asset_returns or {}),
            "yield_calendar": {"yield_asof": row.yield_asof.date().isoformat(),
                               "move_asof": row.move_asof.date().isoformat()},
            "factors": {"dgs10_up_5d": bool(row.dgs10_delta5 > 0),
                        "dgs10": float(dataset.dgs10_publications.dropna().loc[:frame.index[-1]].iloc[-1]),
                        "oas_delta5": float(row.oas_delta5),
                        "oas_used_publication": oas.oas_used_publication.iloc[-1].date().isoformat()},
            "reason": ["insufficient_yield_history"] if stable is None else []}
