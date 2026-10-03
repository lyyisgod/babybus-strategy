"""纯日频因子函数。输入按交易收盘对齐；不联网、不补值、不推断政策。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AUTHOR_LEVELS, STATISTICS, StatisticalConfig, trading_days_since


class DataError(ValueError):
    pass


def validate_frame(frame: pd.DataFrame, core_symbol: str = "SMH") -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        raise DataError("需要非空 DatetimeIndex 日频输入")
    if frame.index.tz is not None or not frame.index.equals(frame.index.normalize()):
        raise DataError("日频索引须为无时区交易日期；可用时间另存元数据")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise DataError("日期重复或未排序")
    prices = {"IXG", "JNK", "QQQ", "SMH", "SPY", "SLV", "TLT", core_symbol}
    required = prices | {f"{s}_adj" for s in prices} | {
        "MOVE", "move_proxy", "VIX", "TNX", "DGS10", "DGS2", "DFF", "BAMLH0A0HYM2"
    }
    missing = required - set(frame.columns)
    if missing:
        raise DataError(f"缺少必需序列: {sorted(missing)}")
    if frame["move_proxy"].isna().any() or not frame["move_proxy"].isin([True, False]).all():
        raise DataError("move_proxy 必须逐日显式标记")
    for column in required - {"move_proxy"}:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise DataError(f"非数值序列: {column}")
        values = frame[column].dropna()
        if not np.isfinite(values).all():
            raise DataError(f"非有限值: {column}")
        if column in prices or column.endswith("_adj") or column in {"MOVE", "VIX", "TNX"}:
            if (values <= 0).any():
                raise DataError(f"价格/指数必须为正: {column}")
    # DGS10另按公布日检查；国债全天休市不能要求股票收盘日有利率。
    if frame[list(required - {"move_proxy", "DGS10"})].iloc[-1].isna().any():
        raise DataError("快照日必需序列缺失；禁止前值填充或回退日期")


def streak(condition: pd.Series) -> pd.Series:
    good = condition.fillna(False).astype(bool)
    return good.astype(int).groupby((~good).cumsum()).cumsum()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder；平盘50、单边涨100、单边跌0；缺口后重新暖机。"""
    changes = close.diff().to_numpy()
    values = np.full(len(close), np.nan)
    gains, losses = [], []
    up = down = None
    for i, change in enumerate(changes):
        if not np.isfinite(change):
            gains, losses, up, down = [], [], None, None
            continue
        gain, loss = max(change, 0), max(-change, 0)
        if up is None:
            gains.append(gain)
            losses.append(loss)
            if len(gains) < period:
                continue
            up, down = float(np.mean(gains)), float(np.mean(losses))
        else:
            up = (up * (period - 1) + gain) / period
            down = (down * (period - 1) + loss) / period
        values[i] = 50 if up == down == 0 else 100 if down == 0 else 100 - 100 / (1 + up / down)
    return pd.Series(values, index=close.index)


def hike_beta(close: pd.Series, yields: pd.Series, days: int = 60) -> pd.Series:
    """收益小数 / 利率百分点；带截距OLS斜率，零方差保持缺失。"""
    returns = close.pct_change(fill_method=None)
    # 60仍是股票交易日窗口；只用其中有当日公布变化的配对样本。
    delta = yields.dropna().diff().reindex(close.index)
    paired_returns = returns.where(delta.notna())
    delta = delta.where(returns.notna())
    beta = paired_returns.rolling(days, min_periods=2).cov(delta) / \
        delta.rolling(days, min_periods=2).var().replace(0, np.nan)
    full_window = returns.rolling(days).count() >= days
    first = yields.dropna().first_valid_index()
    if first is None:
        return beta * np.nan
    coverage = pd.Series(close.index >= first, index=close.index).rolling(days).sum() >= days
    return beta.where(full_window & coverage)


def dip_units(return_1d: float, return_5d: float) -> float:
    if not np.isfinite([return_1d, return_5d]).all():
        raise DataError("加仓收益缺失")
    if return_5d > 0.08 or return_1d >= -0.01:
        return 0.0
    if return_1d >= -0.03:
        return 0.25
    if return_1d >= -0.06:
        return 0.5
    return 1.0


def leap_history(qqq_ret_1d: pd.Series, cooldown_sessions: int = 20) -> pd.Series:
    """按真实交易日冷却；触发后的第1..20日禁止重发，第21日可再触发。"""
    labels = pd.Series(False, index=qqq_ret_1d.index)
    last = None
    for timestamp, value in qqq_ret_1d.items():
        on = timestamp.date()
        if pd.notna(value) and value <= -0.02 and (
            last is None or trading_days_since(last, on) > cooldown_sessions
        ):
            labels.loc[timestamp] = True
            last = on
    return labels


def numeric_factors(frame: pd.DataFrame, core_symbol: str = "SMH",
                    cfg: StatisticalConfig = STATISTICS, *,
                    dgs10_publications=None, move_publications=None) -> pd.DataFrame:
    validate_frame(frame, core_symbol)
    from .yield_calendar import publications, yield_factors
    dgs10 = frame.DGS10 if dgs10_publications is None else dgs10_publications
    move = frame.MOVE if move_publications is None else move_publications
    yield_out = yield_factors(frame.index, dgs10, move, cfg)
    observed_yields = publications(dgs10, frame.index[-1])
    out = pd.DataFrame(index=frame.index)
    for symbol in {"JNK", "QQQ", "SMH", "SLV", "TLT", core_symbol}:
        close = frame[f"{symbol}_adj"]
        out[f"{symbol}_r1"] = close.pct_change(fill_method=None)
        out[f"{symbol}_r5"] = close.pct_change(5, fill_method=None)
        out[f"{symbol}_beta"] = hike_beta(close, observed_yields, cfg.beta_days)
        out[f"{symbol}_rsi14"] = rsi(close)
        out[f"{symbol}_new_low20"] = close < close.shift().rolling(20).min()
    out["JNK_down_streak"] = streak(out.JNK_r1 < 0)
    out["oas_delta5"] = frame.BAMLH0A0HYM2.diff(5)
    out["relative_ixg"] = frame.IXG_adj / frame.SPY_adj
    out["relative_ma200"] = out.relative_ixg.rolling(cfg.relative_ma_days).mean()
    # 不拼接MOVE和TLT波动；每段代理身份变化后重新累计至少252根。
    segments = frame.move_proxy.ne(frame.move_proxy.shift()).cumsum()
    out["move_q80"] = frame.MOVE.groupby(segments).transform(
        lambda s: s.rolling(cfg.move_window, min_periods=cfg.move_min_periods).quantile(cfg.move_quantile))
    out["move_pct5y"] = frame.MOVE.groupby(segments).transform(
        lambda s: s.rolling(cfg.move_window, min_periods=cfg.move_min_periods).apply(
            lambda a: np.nan if np.isnan(a[-1]) else
            100 * ((a < a[-1]).sum() + 0.5 * (a == a[-1]).sum()) / np.isfinite(a).sum(), raw=True))
    out["move_observations"] = frame.MOVE.groupby(segments).transform(
        lambda s: s.rolling(cfg.move_window, min_periods=0).count())
    out = pd.concat([out, yield_out], axis=1)
    out["vix_ma20"] = frame.VIX.rolling(20).mean()
    out["vix_delta1"] = frame.VIX.diff()
    out["vix_q80"] = frame.VIX.rolling(cfg.vix_window).quantile(cfg.vix_quantile)
    out["vix_spike"] = (frame.VIX > out.vix_ma20 * 1.25) | (out.vix_delta1 > 3)
    return out


def author_levels(frame: pd.DataFrame, levels=AUTHOR_LEVELS) -> dict:
    on = frame.index[-1].date()
    result = {}
    for symbol, level in levels.items():
        proxy = symbol == "MOVE" and bool(frame.move_proxy.iloc[-1])
        value = frame[symbol].iloc[-1] if symbol in frame and not proxy else None
        available = value is not None and pd.notna(value)
        result[symbol] = {
            "level": level.value, "unit": level.unit, "as_of": level.as_of.isoformat(),
            "valid_sessions": level.valid_sessions, "age_sessions": trading_days_since(level.as_of, on),
            "expired": level.expired(on),
            "not_yet_effective": on < level.as_of, "proxy": proxy,
            "close": float(value) if available else None,
            "distance": float(value - level.value) if available else None,
            "distance_pct": float(value / level.value - 1) if available else None,
        }
    return result
