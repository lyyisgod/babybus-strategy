"""Reuse the frozen September 21 intraday-tail model on September 25 saved bars."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = PROJECT / "研究记录/2026-09-25-二十交易日建仓/raw"
ORIGINAL = PROJECT / "研究记录/2026-09-21-隔夜至次日爆发"
PROTOCOL = json.loads((HERE / "screen-protocol.json").read_text())
OLD = json.loads((ORIGINAL / "protocol.json").read_text())
AVAILABLE = {p.stem for p in SOURCE.glob("*.json")}
STOCKS = [s for s in OLD["universe"].split() if s in AVAILABLE]
MISSING = [s for s in OLD["universe"].split() if s not in AVAILABLE]
PROXIES = [s for s in OLD["proxies"].split() if s in AVAILABLE]
assert len(STOCKS) == 70 and len(MISSING) == 35
assert {"QQQ", "SMH", "IWM", "JNK", "TLT", "BZ=F", "^VIX"} <= set(PROXIES)

link = HERE / "raw"
if link.is_symlink():
    link.unlink()
link.mkdir(exist_ok=True)
for source_file in SOURCE.glob("*.json"):
    target = link / f"{source_file.stem}-1d.json"
    if not target.exists():
        target.symlink_to(source_file)

# Two September 25 proxy opens exceed vendor highs. Their model inputs use
# closes only; repair the high bound locally so the existing OHLC guard works.
repairs = []
for symbol in ["JNK", "BZ=F"]:
    original_path = SOURCE / f"{symbol}.json"
    target_path = link / f"{symbol}-1d.json"
    data = json.loads(original_path.read_text())
    chart = data["chart"]["result"][0]
    quote = chart["indicators"]["quote"][0]
    idx = -1
    old_high = quote["high"][idx]
    quote["high"][idx] = max(old_high, quote["open"][idx], quote["close"][idx])
    assert quote["high"][idx] > old_high
    if target_path.is_symlink():
        target_path.unlink()
    target_path.write_text(json.dumps(data))
    repairs.append({"symbol": symbol, "field": "2026-09-25 high",
                    "old": old_high, "new": quote["high"][idx],
                    "raw_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                    "reason": "OHLC integrity; downstream model proxy uses closes only"})
(HERE / "input-repairs.json").write_text(json.dumps(repairs, indent=2))

spec = importlib.util.spec_from_file_location("frozen_intraday_model", ORIGINAL / "model.py")
assert spec and spec.loader
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)
model.ROOT = HERE
model.DAY = pd.Timestamp("2026-09-25", tz=model.NY)
model.STOCKS = STOCKS
model.PROXIES = PROXIES
model.quote = lambda symbol, daily: {"symbol": symbol, "last": None, "last_at": None,
    "session": "overnight quote fetched separately", "after_hours_change_pct": None,
    "after_hours_volume": None, "quote_note": "Historical daily-only source; not a live bid/ask"}
(HERE / "pool.json").write_text(json.dumps({"included": STOCKS, "missing_from_original_105": MISSING,
    "source_raw": str(SOURCE), "frozen_protocol": PROTOCOL}, ensure_ascii=False, indent=2))
model.main()
