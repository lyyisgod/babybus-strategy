"""Yahoo 日线/FRED 原频率 parquet 缓存；不补值、不替换证券。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DAILY_FRED, FRED_SYMBOLS, STATISTICS, VERSION, YAHOO_SYMBOLS
from .factors import DataError, validate_frame


@dataclass
class Dataset:
    frame: pd.DataFrame
    payems: pd.Series
    metadata: dict
    dgs10_publications: pd.Series | None = None
    move_publications: pd.Series | None = None


def normalize_index(index):
    index = pd.DatetimeIndex(index)
    # Yahoo 日线日期本身是交易所本地session标签，不能转UTC后再截日。
    return index.tz_localize(None).normalize() if index.tz is not None else index.normalize()


def parse_fred_csv(text: str, symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(StringIO(text), na_values=["."])
    if list(raw.columns) != ["observation_date", symbol]:
        raise DataError(f"FRED 返回了错误序列/格式: {symbol}")
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.pop("observation_date"), errors="raise"))
    raw[symbol] = pd.to_numeric(raw[symbol], errors="raise")
    _check_index(raw)
    return raw


def _check_index(frame):
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        raise DataError("数据序列为空或日期无效")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise DataError("源序列日期重复或未排序")


def yahoo_fetch(symbol: str, start: str, end: str):
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    # identity验证失败必须向外抛出，不用任何其他ticker补位。
    long_name = ticker.get_info().get("longName") if symbol == "IXG" else None
    if symbol == "IXG" and (not isinstance(long_name, str) or "Global Financials" not in long_name):
        raise DataError(f"IXG 身份校验失败: longName={long_name!r}")
    raw = ticker.history(start=start, end=end, interval="1d", auto_adjust=False,
                         actions=False, repair=False, keepna=True, raise_errors=True)
    required = {"Close", "Adj Close"} if not symbol.startswith("^") else {"Close"}
    if raw.empty or not required.issubset(raw.columns):
        raise DataError(f"Yahoo 缺少必需日线列: {symbol}")
    result = raw[sorted(required)].copy()
    result.index = normalize_index(result.index)
    _check_index(result)
    return result, {"source": "yfinance/Yahoo", "source_symbol": symbol,
                    "url": f"https://finance.yahoo.com/quote/{symbol}/history/",
                    "long_name": long_name, "frequency": "daily_close",
                    "unit": "index_points" if symbol.startswith("^") else "USD",
                    "price_basis": "Close=Yahoo_split_adjusted; Adj Close=total_return_adjusted",
                    "pit_verified": False}


def fred_fetch(symbol: str, start: str, end: str):
    response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                            params={"id": symbol, "cosd": start, "coed": end}, timeout=30)
    response.raise_for_status()
    result = parse_fred_csv(response.text, symbol)
    return result, {"source": "FRED latest/revised observations", "source_symbol": symbol,
                    "url": f"https://fred.stlouisfed.org/series/{symbol}",
                    "frequency": "monthly" if symbol == "PAYEMS" else "daily",
                    "unit": "thousands_of_persons" if symbol == "PAYEMS" else "percent",
                    "pit_verified": False,
                    "available_at": None}  # 抓取时间不冒充真实公布时间。


def load_series(symbol, start, end, cache_dir, fetcher, refresh=False,
                allow_stale_cache=False):
    cache_dir = Path(cache_dir)
    key = sha256(f"v1|{symbol}|{start}|{end}".encode()).hexdigest()[:24]
    data_path, meta_path = cache_dir / f"{key}.parquet", cache_dir / f"{key}.json"

    def read_cache():
        meta = json.loads(meta_path.read_text())
        if meta.get("source_symbol") != symbol or meta.get("request") != {"start": start, "end": end}:
            raise DataError(f"缓存序列/请求身份不匹配: {symbol}")
        if meta.get("sha256") != sha256(data_path.read_bytes()).hexdigest():
            raise DataError(f"parquet缓存摘要不匹配: {symbol}")
        if symbol == "IXG" and "Global Financials" not in str(meta.get("long_name", "")):
            raise DataError("IXG 缓存身份校验失败")
        if symbol != "^MOVE" and meta.get("proxy", False):
            raise DataError(f"只有MOVE允许代理，缓存代理被拒绝: {symbol}")
        frame = pd.read_parquet(data_path)
        _check_index(frame)
        return frame, {**meta, "cached": True}

    if not refresh and data_path.exists() and meta_path.exists():
        return read_cache()
    try:
        frame, meta = fetcher(symbol, start, end)
        _check_index(frame)
        if meta.get("source_symbol") != symbol:
            raise DataError(f"下载器返回了错误序列: {symbol}")
        if symbol == "IXG" and "Global Financials" not in str(meta.get("long_name", "")):
            raise DataError("IXG 下载身份校验失败")
        if symbol != "^MOVE" and meta.get("proxy", False):
            raise DataError(f"只有MOVE允许代理，下载代理被拒绝: {symbol}")
    except DataError:
        # 身份/格式错误不能通过陈旧缓存掩盖。
        raise
    except Exception as error:
        if symbol != "^MOVE" or not allow_stale_cache or not data_path.exists() or not meta_path.exists():
            raise DataError(f"拉取失败且无获准缓存: {symbol}: {error}") from error
        frame, meta = read_cache()
        return frame, {**meta, "proxy": True, "stale_cache": True, "fetch_error": str(error)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp = data_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temp)
    temp.replace(data_path)
    meta = {**meta, "proxy": False, "cached": False, "stale_cache": False,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "observation_start": frame.index[0].date().isoformat(),
            "observation_end": frame.index[-1].date().isoformat(),
            "request": {"start": start, "end": end}, "sha256": sha256(data_path.read_bytes()).hexdigest()}
    meta_temp = meta_path.with_suffix(".json.tmp")
    meta_temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    meta_temp.replace(meta_path)
    return frame, meta


def _calendar(start, end):
    import exchange_calendars as xcals
    return xcals.get_calendar("XNYS", start=start, end=end)


def require_closed_session(on, now=None):
    """不把盘中日线、周末或未来日期伪装成完整收盘。"""
    on = pd.Timestamp(on).normalize()
    if on.tz is not None:
        raise DataError("--date 必须是 YYYY-MM-DD 交易日期")
    cal = _calendar(on - pd.Timedelta(days=10), on + pd.Timedelta(days=10))
    if not cal.is_session(on):
        raise DataError(f"不是NYSE交易收盘日: {on.date()}")
    now = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if now.tz is None:
        raise DataError("now 需要时区")
    if now < cal.session_close(on):
        raise DataError("目标日期尚未收盘；不输出盘中日频快照")
    return on


def assemble(yahoo, fred, on, sessions, metadata, core_symbol="SMH", cfg=STATISTICS):
    """NYSE日频坐标；所有缺口原样保留。PAYEMS原频率独立保存。"""
    daily = pd.DataFrame(index=normalize_index(sessions))
    mandatory = (set(YAHOO_SYMBOLS) - {"^MOVE"}) | {core_symbol} | set(FRED_SYMBOLS)
    for symbol in mandatory:
        if metadata.get(symbol, {}).get("proxy", False):
            raise DataError(f"只有MOVE允许代理，必需序列代理被拒绝: {symbol}")
    for symbol in set(YAHOO_SYMBOLS) - {"^MOVE"} | {core_symbol}:
        if symbol not in yahoo:
            raise DataError(f"缺少必需序列: {symbol}")
        source = yahoo[symbol]
        name = {"^VIX": "VIX", "^TNX": "TNX"}.get(symbol, symbol)
        if "Close" not in source:
            raise DataError(f"缺少Close: {symbol}")
        daily[name] = source.Close.reindex(daily.index)
        if not symbol.startswith("^"):
            if "Adj Close" not in source:
                raise DataError(f"缺少复权收益序列: {symbol}")
            daily[f"{name}_adj"] = source["Adj Close"].reindex(daily.index)
    # MOVE历史短不等于缺失：保留真实MOVE，分位不足252根由因子层关闭。
    # 只有缺当日、已有观测区间内缺口、或下载失败代理标记才整段换TLT。
    move = yahoo.get("^MOVE", pd.DataFrame(columns=["Close"])).Close.reindex(daily.index)
    first = move.first_valid_index()
    use_proxy = bool(first is None or pd.isna(move.iloc[-1])
                     or move.loc[first:].isna().any() or metadata.get("^MOVE", {}).get("proxy", False))
    if use_proxy:
        daily["MOVE"] = np.log(daily.TLT_adj / daily.TLT_adj.shift()).rolling(20).std(ddof=1) * np.sqrt(252) * 100
        reason = metadata.get("move_fetch_error", "MOVE missing/incomplete observed sequence")
        metadata["MOVE"] = {"source": "TLT 20-day annualized realized volatility", "source_symbol": "TLT",
                            "proxy": True, "move_proxy": True, "unit": "annualized_percent",
                            "price_basis": "total_return_adjusted", "window_days": 20,
                            "url": "https://finance.yahoo.com/quote/TLT/history/",
                            "fetched_at": metadata.get("TLT", {}).get("fetched_at"),
                            "input_sha256": metadata.get("TLT", {}).get("sha256"),
                            "observation_start": daily["MOVE"].first_valid_index().date().isoformat()
                            if daily["MOVE"].first_valid_index() is not None else None,
                            "observation_end": daily["MOVE"].last_valid_index().date().isoformat()
                            if daily["MOVE"].last_valid_index() is not None else None,
                            "proxy_reason": reason}
    else:
        daily["MOVE"] = move
        metadata["MOVE"] = {**metadata.get("^MOVE", {}), "move_proxy": False}
    daily["move_proxy"] = bool(use_proxy)
    for symbol in FRED_SYMBOLS:
        if symbol not in fred or symbol not in fred[symbol] or fred[symbol][symbol].dropna().empty:
            raise DataError(f"缺少必需FRED序列: {symbol}")
        if symbol in DAILY_FRED:
            daily[symbol] = fred[symbol][symbol].reindex(daily.index)
    # 作者加仓点仅展示，不加入核心状态机的数据依赖。
    for symbol in ("AVGO", "BWXT", "APP"):
        if symbol in yahoo:
            daily[symbol] = yahoo[symbol].Close.reindex(daily.index)
    daily = daily.loc[:pd.Timestamp(on)]
    validate_frame(daily, core_symbol)
    from .yield_calendar import validate_dgs10
    dgs10_publications = fred["DGS10"]["DGS10"].loc[:pd.Timestamp(on)].copy()
    validate_dgs10(dgs10_publications, daily.index)
    move_publications = (daily.MOVE.copy() if use_proxy else
                         yahoo["^MOVE"].Close.loc[:pd.Timestamp(on)].copy())
    payems = fred["PAYEMS"]["PAYEMS"].loc[:pd.Timestamp(on)]
    if payems.dropna().empty:
        raise DataError("PAYEMS目标日期之前无观测；不使用未来观测")
    warnings = ["Yahoo/FRED latest revised observations: not a point-in-time backtest; historical available_at unknown",
                "PAYEMS is monthly in thousands_of_persons; no daily forward-fill, no policy inference",
                "FRED OAS may have only three years; MOVE quantile uses up to five years, minimum 252 observations"]
    if use_proxy:
        warnings.append("move_proxy=true: author MOVE/IXG combo disabled; percentile leg only")
    for symbol in ("AVGO", "BWXT", "APP"):
        if symbol not in daily or pd.isna(daily[symbol].iloc[-1]):
            warnings.append(f"author_levels[{symbol}] display price unavailable")
    gaps = {s: int(daily[s].isna().sum()) for s in DAILY_FRED}
    metadata = {"series": metadata, "warnings": warnings, "fred_daily_missing_sessions": gaps,
                "proxy": use_proxy or any(v.get("proxy", False) for v in metadata.values() if isinstance(v, dict)),
                "move_proxy": use_proxy, "calendar": "XNYS", "alignment": "reindex_without_fill",
                "pit_verified": False, "payems_observation_period": payems.dropna().index[-1].date().isoformat(),
                "yield_factor_version": VERSION,
                "yield_alignment": "publication_results_merge_asof_backward_without_yield_fill"}
    return Dataset(daily, payems, metadata, dgs10_publications, move_publications)


def fetch_dataset(on, cache_dir="data/macro_regime/cache", core_symbol="SMH",
                  refresh=False, allow_stale_cache=False, now=None):
    on = require_closed_session(on, now)
    import yfinance as yf
    yf.set_tz_cache_location(str(Path(cache_dir) / "yfinance"))
    start = (on - pd.DateOffset(years=6)).date().isoformat()
    end = (on + pd.Timedelta(days=1)).date().isoformat()  # Yahoo end为exclusive。
    yahoo, fred, meta = {}, {}, {}
    for symbol in sorted(set(YAHOO_SYMBOLS) | {core_symbol, "AVGO", "BWXT", "APP"}):
        try:
            yahoo[symbol], meta[symbol] = load_series(symbol, start, end, cache_dir, yahoo_fetch,
                                                     refresh, allow_stale_cache)
        except DataError as error:
            if symbol == "^MOVE":
                meta["move_fetch_error"] = str(error)
            elif symbol in ("AVGO", "BWXT", "APP") and symbol != core_symbol:
                meta[symbol] = {"source_symbol": symbol, "proxy": False, "unavailable": True, "fetch_error": str(error)}
            else:
                raise
    for symbol in FRED_SYMBOLS:
        fred[symbol], meta[symbol] = load_series(symbol, start, end, cache_dir, fred_fetch,
                                               refresh, allow_stale_cache)
    sessions = _calendar(start, end).sessions_in_range(start, on)
    return assemble(yahoo, fred, on, sessions, meta, core_symbol)
