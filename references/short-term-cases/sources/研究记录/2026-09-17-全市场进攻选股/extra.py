"""Add three candidates identified from dated same-day primary-source catalysts."""
import importlib.util,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
p=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('scan',p/'scan.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
symbols=['GNRC','MTSI','TSEM']
jobs=[(s,k) for s in symbols for k in ['daily','intraday']]
with ThreadPoolExecutor(max_workers=3) as ex:manifest=list(ex.map(m.core.fetch,jobs))
(p/'extra-manifest.json').write_text(json.dumps({'asof':m.NOW.isoformat(),'reason':'New candidates identified in dated primary-source news; added after initial scan.','requests':manifest},indent=2))
rows=[]
for s in symbols:
 d=m.core.history(s)[0];z=m.core.factors(d).iloc[-1].to_dict();z.pop('price');q=m.intraday(s,d)
 rows.append({**z,**q})
pd.DataFrame(rows).to_csv(p/'extra-factors.csv',index=False)
print(pd.DataFrame(rows)[['symbol','price','time_et','change_pct','ret20','atr14','ema20','ema50','same_time_volume_ratio']].round(3).to_string(index=False))
