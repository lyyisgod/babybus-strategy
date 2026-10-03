import json, urllib.request, ssl, certifi, concurrent.futures, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
def fetch(job):
    key,url=job
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=30,context=ssl.create_default_context(cafile=certifi.where())) as f: body=f.read().decode()
        at=datetime.datetime.now(NY).isoformat()
        if key.endswith('.csv'):
            (ROOT/key).write_text(body)
            return {'key':key,'source':url,'retrieved_at':at}
        data=json.loads(body)['chart']['result'][0]
        (ROOT/(key+'.json')).write_text(json.dumps({'source':url,'retrieved_at':at,'data':data}))
        return {'key':key,'time':data['meta'].get('regularMarketTime'),'price':data['meta'].get('regularMarketPrice'),'bars':len(data.get('timestamp',[]))}
    except Exception as e: return {'key':key,'error':str(e)}
jobs=[]
for s in ['MU','SMH','QQQ','SNDK','WDC','STX','NVDA','^VIX']:
    for suffix,rng,interval in [('daily','2y','1d'),('intraday','1d','1m')]:
        jobs.append((s.replace('^','')+'_'+suffix,f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range={rng}&interval={interval}&includePrePost=false&events=div%2Csplits'))
for s in ['MU','SMH','QQQ']:
    jobs.append((s+'_5m',f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=60d&interval=5m&includePrePost=false'))
for s in ['DFII10','BAMLH0A0HYM2','DCOILBRENTEU','CPILFESL']:
    jobs.append((s+'.csv',f'https://fred.stlouisfed.org/graph/graph.csv?id={s}&cosd=2025-01-01'))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    result=list(pool.map(fetch,jobs))
(ROOT/'fetch_status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
