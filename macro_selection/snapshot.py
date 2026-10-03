"""实时收盘快照入口；缺必需数据时输出显式失败仪表并 exit 2。

python -m macro_selection.snapshot --official-evidence FILE --calendars FILE
    [--oas-publications FILE] [--candidates FILE] [--gross-exposure NUMBER]
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import exchange_calendars as xc

from macro_regime.config import VERSION, DAILY_FRED
from macro_regime.data import fetch_dataset
from macro_regime.factors import DataError
from macro_regime.yield_calendar import treasury_closed
from .bridge import macro_output, instrument_rules
from .policy import official_regime
from .selection import select, value, finite


def latest_closed_nyse(now):
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise DataError("runtime requires a timezone")
    day = now.tz_convert("America/New_York").normalize().tz_localize(None)
    cal = xc.get_calendar("XNYS", start=day - pd.Timedelta(days=30), end=day + pd.Timedelta(days=10))
    sessions = cal.sessions_in_range(day - pd.Timedelta(days=20), day)
    return next(s for s in reversed(sessions) if cal.session_close(s) <= now)


def cache_audit(cache, on, *, request_start, request_end):
    facts = {}
    for path in sorted(Path(cache).glob("*.json")):
        m = json.loads(path.read_text())
        if m.get("request") != {"start": request_start, "end": request_end}:
            continue
        symbol = m["source_symbol"]
        raw = pd.read_parquet(path.with_suffix(".parquet")).loc[:on]
        s = raw[symbol] if symbol in raw else raw.Close
        s = s.dropna()
        facts[symbol] = {"series": symbol, "source": m["source"], "url": m["url"],
                         "observation_date": None if s.empty else s.index[-1].date().isoformat(),
                         "latest_value": None if s.empty else float(s.iloc[-1]), "unit": m["unit"],
                         "snapshot_date": on.date().isoformat(),
                         "snapshot_value": None if on not in s else float(s.loc[on]),
                         "fetched_at": m["fetched_at"], "sha256": m["sha256"], "proxy": m["proxy"],
                         "publication_dates_verified": False, "pit_verified": False}
    return facts


def run(now, evidence, calendars, cache, *, oas_record=None, candidate_loader=None, gross=None):
    on = latest_closed_nyse(now)
    date = on.date().isoformat()
    policy = official_regime(evidence.get("statements", []), evidence.get("next_meeting_probs"), date)
    base = {"version": VERSION, "asof": date, "runtime": pd.Timestamp(now).isoformat(),
            **policy, "last_fomc": evidence.get("statements", []),
            "next_meeting_probs": evidence.get("next_meeting_probs"),
            "stress": None, "divergence": None, "yield_stable": None,
            "balance_sheet_stress": None,
            "constraints": {"allow_margin": None, "max_gross": None, "allow_new_core": False,
                            "allow_new_satellite": False}, "data_valid": False}
    start, end = (on - pd.DateOffset(years=6)).date().isoformat(), (on + pd.Timedelta(days=1)).date().isoformat()
    try:
        dataset = fetch_dataset(on, cache_dir=cache, refresh=True, now=now)
        if oas_record is None:
            raise DataError("OAS publication schedule unavailable; cannot assert one-publication lag")
        oas = pd.Series([x["value"] for x in oas_record["observations"]],
                        index=pd.to_datetime([x["publication_date"] for x in oas_record["observations"]]))
        base = {**base, **macro_output(dataset, policy, oas_publications=oas,
                                      oas_metadata=oas_record)}
    except (DataError, ValueError, TypeError, OSError) as error:
        base["errors"] = [str(error)]
    base["sources"] = cache_audit(cache, on, request_start=start, request_end=end)
    missing = [k for k in DAILY_FRED if base["sources"].get(k, {}).get("snapshot_value") is None
               and not (k == "DGS10" and treasury_closed(on))]
    if missing:
        base.setdefault("errors", []).append("snapshot_observation_missing: " + ",".join(missing))
    # 不在宏观失败/政策封锁时读取候选、历史 EPS 或编造一个股票池。
    candidates = []
    if base["data_valid"] and base["regime"] in {"HIKING", "PAUSE"} and candidate_loader is not None:
        candidates = candidate_loader()
        returns = {c["symbol"]: {"asof": c["asof"],
                                 "ret_1d": value(c, "ret_1d"), "ret_5d": value(c, "ret_5d")}
                   for c in candidates}
        # 冻结的序数计算在桥接层完成；选择层只读取输出。
        valid_returns = {s: r for s, r in returns.items() if finite(r["ret_1d"]) and finite(r["ret_5d"])}
        base["asset_rules"] = instrument_rules(valid_returns)
    result = select(base, candidates, calendars=calendars, gross_exposure=gross)
    if not candidates and base["data_valid"] and base["regime"] in {"HIKING", "PAUSE"}:
        result["selection_missing"] = "no_verified_candidate_input"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="运行时最新已收盘NYSE日期；只读研究选择层")
    parser.add_argument("--official-evidence", required=True, type=Path)
    parser.add_argument("--calendars", required=True, type=Path)
    parser.add_argument("--oas-publications", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--gross-exposure", type=float)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/macro_selection/cache"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence, calendars = (json.loads(p.read_text()) for p in (args.official_evidence, args.calendars))
    oas_record = json.loads(args.oas_publications.read_text()) if args.oas_publications else None
    loader = (lambda: json.loads(args.candidates.read_text())) if args.candidates else None
    result = run(datetime.now(timezone.utc), evidence, calendars, args.cache_dir,
                 oas_record=oas_record, candidate_loader=loader, gross=args.gross_exposure)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)
    return 2 if not result["macro"]["data_valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
