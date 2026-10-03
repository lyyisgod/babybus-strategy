from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import urllib.request, ssl, certifi, hashlib, json, sys

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
def fetch(job):
    symbol,interval=job;span='10y' if interval=='1d' else '5d'
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={span}&interval={interval}&includePrePost=true&events=div%2Csplits'
    try:
        raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research/1.0'}),timeout=25,context=ssl.create_default_context(cafile=certifi.where())).read()
        r=json.loads(raw)['chart']['result'][0];assert r['timestamp']
        (ROOT/'raw'/f'{symbol}-{interval}.json').write_bytes(raw)
        return dict(symbol=symbol,interval=interval,url=url,fetched_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest())
    except Exception as e:return dict(symbol=symbol,interval=interval,url=url,error=str(e))
if __name__=='__main__':
    (ROOT/'raw').mkdir(exist_ok=True)
    symbols=list(dict.fromkeys((P['universe']+' '+P['proxies']).split()))
    jobs=[(s,k) for s in symbols for k in ['1d','5m']]
    if '--quotes-only' in sys.argv:jobs=[(s,'5m') for s in sys.argv[sys.argv.index('--quotes-only')+1:]]
    with ThreadPoolExecutor(max_workers=6) as ex:rows=list(ex.map(fetch,jobs))
    name='refresh-manifest.json' if '--quotes-only' in sys.argv else 'manifest.json'
    (ROOT/name).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print(json.dumps({'requests':len(rows),'successful':sum('error' not in r for r in rows),'errors':[r for r in rows if 'error' in r]},ensure_ascii=False))
