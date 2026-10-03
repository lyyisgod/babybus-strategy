import json, ssl, urllib.request, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import certifi

ROOT=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
NOW=datetime.now(timezone.utc)
OFFLINE='--offline' in sys.argv
if OFFLINE:
    NOW=datetime.fromisoformat(json.loads((ROOT/'snapshot.json').read_text())['asof'])
SYMBOLS='NVDA AMD MU AVGO TSM MRVL GOOGL META MSFT AAPL AMZN ORCL CRM NOW CRWD PANW DDOG PLTR XOM CVX JPM WMT COST SMH SOXX QQQ SPY IGV'.split()
CTX=ssl.create_default_context(cafile=certifi.where())
def fetch(s):
    try:
        url=f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=5d&interval=5m&includePrePost=true'
        if OFFLINE:
            raw=json.loads((ROOT/f'{s}-intraday.json').read_text())
        else:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),context=CTX,timeout=20) as r: raw=json.load(r)
            (ROOT/f'{s}-intraday.json').write_text(json.dumps(raw))
        x=raw['chart']['result'][0]; m=x['meta']; q=x['indicators']['quote'][0]
        rows=[]
        for i,t in enumerate(x['timestamp']):
            dt=datetime.fromtimestamp(t,NY)
            if dt.date()!=NOW.astimezone(NY).date() or t>NOW.timestamp() or q['close'][i] is None: continue
            rows.append(dict(time=dt.isoformat(),stamp=t,**{k:q[k][i] for k in ['open','high','low','close','volume']}))
        reg=[b for b in rows if '09:30'<=b['time'][11:16]<'16:00']
        post=[b for b in rows if b['time'][11:16]>='16:00']
        close=m['regularMarketPrice']; hi=m.get('regularMarketDayHigh'); lo=m.get('regularMarketDayLow')
        # chartPreviousClose is the previous close before the five-day range; use previousClose if present.
        prev=m.get('previousClose')
        if prev is None:
            dates=sorted(set(datetime.fromtimestamp(t,NY).date() for t in x['timestamp'] if datetime.fromtimestamp(t,NY).date()<NOW.astimezone(NY).date()))
            prevrows=[q['close'][i] for i,t in enumerate(x['timestamp']) if dates and datetime.fromtimestamp(t,NY).date()==dates[-1] and '09:30'<=datetime.fromtimestamp(t,NY).strftime('%H:%M')<'16:00' and q['close'][i] is not None]
            prev=prevrows[-1] if prevrows else None
        vr=[b for b in reg if b['volume'] is not None and b['volume']>0 and b['high'] is not None and b['low'] is not None]
        vwap=sum((b['high']+b['low']+b['close'])/3*b['volume'] for b in vr)/sum(b['volume'] for b in vr) if vr else None
        last=post[-1] if post else None
        # 16:00 bar may include the closing auction; do not call it after-hours buying volume.
        positive=[b for b in post if b['time'][11:16]>'16:00' and b['volume'] is not None and b['volume']>0]
        hour=[b for b in reg if b['time'][11:16]>='15:00']
        return dict(symbol=s,source=url,fetched_at=datetime.now(timezone.utc).isoformat(),regular_close=close,regular_at=datetime.fromtimestamp(m['regularMarketTime'],NY).isoformat(),day_pct=100*(close/prev-1) if prev else None,high=hi,low=lo,close_position=100*(close-lo)/(hi-lo) if hi and lo and hi>lo else None,vwap_proxy=vwap,last_hour_pct=100*(close/hour[0]['open']-1) if hour else None,after_price=last['close'] if last else None,after_at=last['time'] if last else None,after_pct=100*(last['close']/close-1) if last else None,after_volume_reported=sum(b['volume'] for b in positive) if positive else None,after_positive_volume_bars=len(positive),post_last=post[-4:])
    except Exception as e: return dict(symbol=s,error=str(e))
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=6) as pool: result=list(pool.map(fetch,SYMBOLS))
    out=dict(asof=NOW.isoformat(),universe=SYMBOLS,rows=result,note='After-hours prices are five-minute bar snapshots, not executable bid/ask. Zero or missing volume does not prove no trading. VWAP is five-minute typical-price proxy.')
    (ROOT/'snapshot.json').write_text(json.dumps(out,indent=2))
    for r in result:
        print(json.dumps({k:v for k,v in r.items() if k not in ['source','post_last','fetched_at']},ensure_ascii=False))
