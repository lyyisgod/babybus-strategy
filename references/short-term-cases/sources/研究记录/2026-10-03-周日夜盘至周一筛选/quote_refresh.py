"""只读刷新周五盘后价格代理及实际利率，保存来源与时点。"""
from pathlib import Path
import sys, json, ssl, hashlib, urllib.request
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import certifi
import pandas as pd

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[1]))
from strategies.bb_washout_bounce.data import save
names=['ALAB','IREN','APLD','STX','WDC','AAOI','COHR','ON']
folder=P/'extended-raw';folder.mkdir(exist_ok=False)

def get(symbol):
    url=f'https://query2.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?range=5d&interval=5m&includePrePost=true'
    at=datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=20,
                                    context=ssl.create_default_context(cafile=certifi.where())) as response:
            raw=response.read()
        path=folder/f'{symbol}.json';path.write_bytes(raw)
        data=json.loads(raw)['chart']['result'][0]
        bars=pd.DataFrame(data['indicators']['quote'][0],index=pd.to_datetime(data['timestamp'],unit='s',utc=True).tz_convert('America/New_York')).dropna(subset=['close'])
        bars=bars[(bars.index.strftime('%Y-%m-%d')=='2026-10-02') & (bars.index.hour>=16) & (bars.index.hour<20)]
        return {'symbol':symbol,'source':url,'fetched_at':at,'sha256':hashlib.sha256(raw).hexdigest(),
                'last_extended_bar_price':float(bars.close.iloc[-1]) if len(bars) else None,
                'last_extended_bar_at':bars.index[-1] if len(bars) else None,
                'regular_market_at':pd.Timestamp(data['meta']['regularMarketTime'],unit='s',tz='UTC'),
                'regular_market_price':data['meta']['regularMarketPrice'],
                'overnight_bid_ask':None,'price_kind':'Yahoo extended 5m bar proxy, not executable ask'}
    except Exception as error:
        return {'symbol':symbol,'source':url,'fetched_at':at,'error':str(error)}

with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(get,names))
save(P/'extended_quotes.json',results)
print(json.dumps(results,default=str,ensure_ascii=False),flush=True)
url='https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10'
with urllib.request.urlopen(url,timeout=20,context=ssl.create_default_context(cafile=certifi.where())) as response:
    raw=response.read()
(P/'DFII10.csv').write_bytes(raw)
save(P/'DFII10-source.json',{'source':url,'fetched_at':datetime.now(timezone.utc).isoformat(),
                          'sha256':hashlib.sha256(raw).hexdigest(),'pit_verified':False})
