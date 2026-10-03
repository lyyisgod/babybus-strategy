#!/usr/bin/env python3
"""Auditable TSLA/TSLL anchor and gap event study.

Inputs are Nasdaq historical JSON files.  A current regular-session bar can be
appended because Nasdaq's historical endpoint may lag the latest close.
The statistics are descriptive and do not claim causal or predictive power.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_nasdaq(path: str) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    rows = payload["data"]["tradesTable"]["rows"]
    clean = []
    for row in rows:
        clean.append(
            {
                "date": pd.to_datetime(row["date"]),
                "open": float(row["open"].replace("$", "").replace(",", "")),
                "high": float(row["high"].replace("$", "").replace(",", "")),
                "low": float(row["low"].replace("$", "").replace(",", "")),
                "close": float(row["close"].replace("$", "").replace(",", "")),
                "volume": float(row["volume"].replace(",", "")),
            }
        )
    return pd.DataFrame(clean).drop_duplicates("date").sort_values("date").set_index("date")


def append_bar(frame: pd.DataFrame, spec: str | None) -> pd.DataFrame:
    if not spec:
        return frame
    date, o, h, l, c, v = spec.split(",")
    frame.loc[pd.Timestamp(date)] = list(map(float, (o, h, l, c, v)))
    return frame.sort_index()


def summary(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {"n": 0}
    return {
        "n": int(values.size),
        "up_rate": round(float((values > 0).mean()), 4),
        "mean_pct": round(float(values.mean() * 100), 3),
        "median_pct": round(float(values.median() * 100), 3),
        "p10_pct": round(float(values.quantile(0.10) * 100), 3),
        "p90_pct": round(float(values.quantile(0.90) * 100), 3),
    }


def event_study(tsla: pd.DataFrame, tsll: pd.DataFrame) -> dict:
    d = tsla.add_prefix("tsla_").join(tsll.add_prefix("tsll_"), how="inner")
    d["tsla_gap_pct"] = d.tsla_open / d.tsla_close.shift(1) - 1
    d["tsll_cross_below_10"] = (d.tsll_close < 10) & (d.tsll_close.shift(1) >= 10)
    d["combined"] = (d.tsll_close < 10) & (d.tsla_gap_pct > 0)
    for horizon in (1, 5, 10):
        d[f"fwd_{horizon}"] = d.tsla_close.shift(-horizon) / d.tsla_close - 1
    result = {}
    masks = {
        "tsll_below_10_all_days": d.tsll_close < 10,
        "tsll_first_close_below_10": d.tsll_cross_below_10,
        "tsll_below_10_and_tsla_gap_up": d.combined,
    }
    # Exclude the current row from the historical event list; forward returns
    # naturally remain NaN, but this also makes the date list explicit.
    history = d.iloc[:-1]
    for label, mask in masks.items():
        selected = history.loc[mask.reindex(history.index, fill_value=False)]
        result[label] = {
            "event_dates": [x.date().isoformat() for x in selected.index],
            **{f"forward_{h}d": summary(selected[f"fwd_{h}"]) for h in (1, 5, 10)},
        }
    return result


def gap_inventory(frame: pd.DataFrame) -> dict:
    rows = []
    for i in range(1, len(frame)):
        prev = frame.iloc[i - 1]
        cur = frame.iloc[i]
        date = frame.index[i]
        later = frame.iloc[i + 1 :]
        if cur.low > prev.high:
            # Up-gap is fully filled only after later trading reaches prev high.
            subsequent_low = float(later.low.min()) if not later.empty else float("inf")
            if subsequent_low > prev.high:
                remaining_top = min(float(cur.low), subsequent_low)
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "direction": "up",
                        "original": [round(float(prev.high), 4), round(float(cur.low), 4)],
                        "remaining": [round(float(prev.high), 4), round(remaining_top, 4)],
                    }
                )
        elif cur.high < prev.low:
            # Down-gap is fully filled only after later trading reaches prev low.
            subsequent_high = float(later.high.max()) if not later.empty else -float("inf")
            if subsequent_high < prev.low:
                remaining_bottom = max(float(cur.high), subsequent_high)
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "direction": "down",
                        "original": [round(float(cur.high), 4), round(float(prev.low), 4)],
                        "remaining": [round(remaining_bottom, 4), round(float(prev.low), 4)],
                    }
                )
    return {"definition": "Full daily-range gaps; partial remainder shown after subsequent fills.", "unfilled": rows}


def technical_state(frame: pd.DataFrame) -> dict:
    c = frame.close
    prior = c.shift(1)
    tr = pd.concat([(frame.high - frame.low), (frame.high - prior).abs(), (frame.low - prior).abs()], axis=1).max(axis=1)
    logret = np.log(c / prior)
    return {
        "date": frame.index[-1].date().isoformat(),
        "close": round(float(c.iloc[-1]), 4),
        "return_1d_pct": round(float((c.iloc[-1] / c.iloc[-2] - 1) * 100), 3),
        "return_5d_pct": round(float((c.iloc[-1] / c.iloc[-6] - 1) * 100), 3),
        "return_20d_pct": round(float((c.iloc[-1] / c.iloc[-21] - 1) * 100), 3),
        "sma_20": round(float(c.tail(20).mean()), 4),
        "sma_50": round(float(c.tail(50).mean()), 4),
        "sma_200": round(float(c.tail(200).mean()), 4),
        "atr_20": round(float(tr.tail(20).mean()), 4),
        "rv_20_annualized_pct": round(float(logret.tail(20).std() * np.sqrt(252) * 100), 3),
        "volume_ratio_20": round(float(frame.volume.iloc[-1] / frame.volume.tail(20).mean()), 3),
        "prior_close": round(float(c.iloc[-2]), 4),
        "opening_gap_pct": round(float((frame.open.iloc[-1] / c.iloc[-2] - 1) * 100), 3),
        "today_range_overlaps_prior_range": bool(frame.low.iloc[-1] <= frame.high.iloc[-2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tsla_json")
    parser.add_argument("tsll_json")
    parser.add_argument("--tsla-current")
    parser.add_argument("--tsll-current")
    args = parser.parse_args()
    tsla = append_bar(load_nasdaq(args.tsla_json), args.tsla_current)
    tsll = append_bar(load_nasdaq(args.tsll_json), args.tsll_current)
    out = {
        "method": "Descriptive event study using Nasdaq daily OHLCV; overlapping below-$10 days are reported separately from first-cross events. No costs or slippage.",
        "tsla_state": technical_state(tsla),
        "tsll_state": technical_state(tsll),
        "event_study": event_study(tsla, tsll),
        "tsla_gap_inventory": gap_inventory(tsla),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
