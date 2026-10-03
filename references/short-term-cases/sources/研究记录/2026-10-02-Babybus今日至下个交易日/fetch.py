from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('prior',Path('研究记录/2026-10-01-夜盘Babybus全面筛选/scan.py').resolve())
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.P=Path(__file__).resolve().parent
m.PROTO=m.json.loads((m.P/'protocol.json').read_text())
m.fetch();m.macrofetch()
