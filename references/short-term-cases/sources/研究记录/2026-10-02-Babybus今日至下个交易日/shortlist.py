from pathlib import Path
import sys,json,datetime,urllib.request,ssl,hashlib
from concurrent.futures import ThreadPoolExecutor
import certifi,pandas as pd,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(ROOT))
from strategies.bb_washout_bounce.data import save,load_config,load_market,calendar
names=['IREN','CIFR','APLD','AAOI','COHR','VICR','RDW','UMAC','CRWV','NBIS','ON','WULF','QQQ','SMH','JNK','SPY']
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),names=names,clock='10:10 America/New_York; completed 10:05 bar close',live_rvol='20 prior full 09:30-10:10 five-minute paths',historical_entry='signal prior completed daily gates >=5; previous 90d RV >=60%; ADV>=50m; same-clock price>VWAP; no fitted coefficients',exit='first +5% limit or -3% stop through next session15:55, next-session15:55 time exit',same_bar_dual_hit='stop first',overnight_gap='adverse stop gap at open; positive take-profit gap capped at target',cost_roundtrip=.002,historical_test='price-only diagnostic/current surviving eight names, reused period/no calibration',current_probability=None)
if not (P/'shortlist-protocol.json').exists():save(P/'shortlist-protocol.json',protocol)
folder=P/'shortlist-raw';folder.mkdir(exist_ok=False)
def one(s):
 url=f'https://query2.finance.yahoo.com/v8/finance/chart/{s}?range=60d&interval=5m&includePrePost=false&events=div%2Csplits'
 row=dict(ticker=s,url=url,fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=25,context=ssl.create_default_context(cafile=certifi.where())) as r:raw=r.read()
  assert json.loads(raw)['chart']['result'][0];fn=s+'.json';(folder/fn).write_bytes(raw);row.update(file=fn,sha256=hashlib.sha256(raw).hexdigest())
 except Exception as e:row['error']=str(e)
 return row
with ThreadPoolExecutor(max_workers=8) as pool:manifest=list(pool.map(one,names))
save(folder/'manifest.json',manifest);print('success',sum('error' not in r for r in manifest),len(names))
