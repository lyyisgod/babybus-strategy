import json,urllib.request,ssl,certifi,concurrent.futures,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
jobs={}
for s in ['AAPL','HOOD','APP','XOM','QQQ']:
 jobs[s+'_5m.json']=f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=60d&interval=5m&includePrePost=false'
for s in ['^VIX','^TNX','BZ=F']:
 jobs[s.replace('^','')+'_live.json']=f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=1d&interval=1m'
def get(x):
 k,u=x
 try:
  r=urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=25,context=ssl.create_default_context(cafile=certifi.where())).read()
  (ROOT/k).write_bytes(r)
  out={'file':k,'url':u,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bytes':len(r)}
  if k.endswith('json'):
   a=json.loads(r)['chart']['result'][0];out['meta']={key:a['meta'].get(key) for key in ['regularMarketPrice','regularMarketTime','chartPreviousClose','previousClose','dataGranularity']};out['bars']=len(a.get('timestamp',[]))
  else:out['tail']=r.decode().splitlines()[-2:]
  return out
 except Exception as e:return {'file':k,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p:r=list(p.map(get,jobs.items()))
(ROOT/'extra_sources.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
