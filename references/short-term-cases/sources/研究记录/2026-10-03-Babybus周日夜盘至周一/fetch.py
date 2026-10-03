from pathlib import Path
import importlib.util
import json

P = Path(__file__).resolve().parent
ROOT = P.parent.parent
spec = importlib.util.spec_from_file_location('prior_scan', ROOT / '研究记录/2026-10-01-夜盘Babybus全面筛选/scan.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.P = P
m.PROTO = json.loads((P / 'protocol.json').read_text())
m.fetch()
m.macrofetch()
