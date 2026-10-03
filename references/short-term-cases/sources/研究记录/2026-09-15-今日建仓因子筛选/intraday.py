"""Read-only intraday verification, completed five-minute bars only."""
import json, ssl, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import certifi

R=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
SYMS='SPY QQQ SMH XLE CIBR IGV XOM CVX COP EOG NOW CRWD FTNT AMD NVDA MU AVGO TSLA TSLL ABBV BRK-B MSFT'.split()
D={x['symbol']:x for x in json.loads((R/'snapshot.json').read_text())}

def fetch(s):
    now=datetime.now(timezone.utc); day=now.astimezone(NY).strftime('%Y-%m-%d')
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?range=5d&interval=5m&includePrePost=false'
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),context=ssl.create_default_context(cafile=certifi.where()),timeout=25) as f: raw=json.load(f)
        (R/'raw'/f'{s}-5m.json').write_text(json.dumps(raw))
        x=raw['chart']['result'][0]; q=x['indicators']['quote'][0];m=x['meta']; rows=[]
        # Use quote timestamp as well as clock to exclude unfinished bars.
        cutoff=min(now.timestamp(),m['regularMarketTime'])
        for i,t in enumerate(x['timestamp']):
            dt=datetime.fromtimestamp(t,NY)
            if not '09:30'<=dt.strftime('%H:%M')<'16:00' or t+300>cutoff:continue
            b={k:q[k][i] for k in ['open','high','low','close','volume']}
            if any(v is None for v in b.values()):continue
            b.update(time=dt.isoformat(),date=dt.strftime('%Y-%m-%d'),hm=dt.strftime('%H:%M'));rows.append(b)
        today=[b for b in rows if b['date']==day]
        if not today:raise ValueError('No completed bars today')
        vol=sum(b['volume'] for b in today);vwap=sum((b['high']+b['low']+b['close'])/3*b['volume'] for b in today)/vol
        hm=today[-1]['hm']; prior=sorted(set(b['date'] for b in rows if b['date']<day))
        pv=[sum(b['volume'] for b in rows if b['date']==d and b['hm']<=hm) for d in prior]
        orb=[b for b in today if b['hm']<'10:00']
        return dict(symbol=s,source=url,retrieved_at=datetime.now(timezone.utc).isoformat(),price=m['regularMarketPrice'],quote_at=datetime.fromtimestamp(m['regularMarketTime'],NY).isoformat(),day_pct=100*(m['regularMarketPrice']/D[s]['prev_close']-1),vwap_proxy=vwap,last_completed_bar=today[-1],last3_close=[b['close'] for b in today[-3:]],above_vwap_last3=all(b['close']>vwap for b in today[-3:]),rvol_same_time_4session=vol/(sum(pv)/len(pv)) if pv and min(pv)>0 else None,prior_sessions=len(prior),orb_high=max(b['high'] for b in orb),orb_low=min(b['low'] for b in orb),day_low=min(b['low'] for b in today),day_high=max(b['high'] for b in today))
    except Exception as e:return dict(symbol=s,error=str(e))

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(fetch,SYMS))
    (R/'intraday.json').write_text(json.dumps(rows,indent=2))
    for r in rows:print(json.dumps(r))
