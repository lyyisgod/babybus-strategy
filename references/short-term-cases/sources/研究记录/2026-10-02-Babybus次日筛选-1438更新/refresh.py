from pathlib import Path
import sys, json, datetime, ssl, urllib.request, hashlib
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
import certifi
P=Path(__file__).resolve().parent; ROOT=P.parents[1]; sys.path.insert(0,str(ROOT))
from strategies.bb_washout_bounce.data import save

def batch(names, folder, span):
    folder.mkdir(exist_ok=False)
    def one(s):
        url=f'https://query2.finance.yahoo.com/v8/finance/chart/{quote(s)}?range={span}&interval=5m&includePrePost=false&events=div%2Csplits'
        row=dict(ticker=s,symbol=s,url=url)
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=25,context=ssl.create_default_context(cafile=certifi.where())) as r: raw=r.read()
            assert json.loads(raw)['chart']['result'][0]
            row['fetched_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            fn=s.replace('^','INDEX_').replace('=','_')+'.json'; (folder/fn).write_bytes(raw)
            row.update(file=fn,sha256=hashlib.sha256(raw).hexdigest())
        except Exception as e:row['error']=str(e)
        return row
    with ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(one,names))
    save(folder/'manifest.json',rows)
    print(folder.name,'success',sum('error' not in r for r in rows),'requested',len(rows),flush=True)

if __name__=='__main__':
    proto=json.loads((P/'protocol.json').read_text())
    batch(proto['stocks']+proto['proxies']+['IXG','^MOVE','NQ=F','TSLL'],P/'live-raw','5d')
    short=json.loads((P/'shortlist-protocol.json').read_text())
    batch(short['names'],P/'shortlist-raw','60d')
    rows=[]
    for s in ['BAMLH0A0HYM2','DFII10']:
        url=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}'
        row=dict(series=s,source=url)
        try:
            with urllib.request.urlopen(url,timeout=25,context=ssl.create_default_context(cafile=certifi.where())) as r:raw=r.read()
            assert raw.startswith(b'DATE,') or raw.startswith(b'observation_date,')
            (P/(s+'.csv')).write_bytes(raw);row.update(sha256=hashlib.sha256(raw).hexdigest(),fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        except Exception as e:row['error']=str(e)
        rows.append(row)
    save(P/'macro-manifest.json',rows);print('macro',rows,flush=True)
