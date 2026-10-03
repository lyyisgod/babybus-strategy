"""Refresh three final contenders; preserve original batch snapshot."""
import importlib.util,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
p=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('scan',p/'scan.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
syms=['CLS','LITE','ALAB']
ds={s:m.core.history(s)[0] for s in syms}
out=p/'latest';(out/'raw').mkdir(parents=True,exist_ok=True);m.core.ROOT=out
with ThreadPoolExecutor(max_workers=3) as ex:manifest=list(ex.map(m.core.fetch,[(s,'intraday') for s in syms]))
(out/'manifest.json').write_text(json.dumps({'asof':m.NOW.isoformat(),'requests':manifest},indent=2))
rows=[m.intraday(s,ds[s]) for s in syms]
pd.DataFrame(rows).to_csv(out/'quotes.csv',index=False)
print(pd.DataFrame(rows)[['symbol','price','change_pct','time_et','vwap_proxy','same_time_volume_ratio']].round(3).to_string(index=False))
