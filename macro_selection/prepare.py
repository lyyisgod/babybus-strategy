"""收盘价格候选事实；不联网，不以新闻补齐一致预期。"""
import pandas as pd
import numpy as np
from macro_regime.factors import DataError, rsi, hike_beta


def price_facts(symbol, prices, asof, source, *, dgs10_publications=None):
    """以统一复权 Close 计算价格比率；不混用复权前后高点。

    author IXG 未复权价仍由原宏观引擎独立处理。本函数只准备候选。
    窗口含 t 及之前 62 根；EPS 基准为该窗口首个交易日。
    """
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise DataError("candidate prices must have sorted unique daily dates")
    close = prices.loc[:asof].copy()
    if not source or len(close) < 200 or close.index[-1].date().isoformat() != asof:
        raise DataError("candidate closed history/source missing")
    if close.isna().any() or not np.isfinite(close).all() or (close <= 0).any():
        raise DataError("candidate price gaps cannot be filled")
    values = {"close": float(close.iloc[-1]),
              "ret_1d": float(close.pct_change(fill_method=None).iloc[-1]),
              "ret_5d": float(close.pct_change(5, fill_method=None).iloc[-1]),
              "drawdown_63d": float(1 - close.iloc[-1] / close.iloc[-63:].max()),
              "below_ma50": bool(close.iloc[-1] < close.iloc[-50:].mean()),
              "below_ma200": bool(close.iloc[-1] < close.iloc[-200:].mean()),
              "rsi14": float(rsi(close).iloc[-1]),
              "new_low20": bool(close.iloc[-1] < close.iloc[-21:-1].min())}
    if dgs10_publications is not None:
        beta = hike_beta(close, dgs10_publications.loc[:asof], 60).iloc[-1]
        if pd.notna(beta):
            values["hike_beta"] = float(beta)
    facts = {k: {"value": v, "date": asof, "series": f"{symbol}.{k}",
                 "source": source, "closed": True, "price_basis": "consistently_adjusted_close"}
             for k, v in values.items()}
    return {"facts": facts, "window_start": close.index[-63].date().isoformat()}
