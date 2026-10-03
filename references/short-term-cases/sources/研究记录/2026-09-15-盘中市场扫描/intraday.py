"""One-session descriptive screen; no probability or backtest claim."""
import json, ssl, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import certifi

OUT = Path(__file__).resolve().parent
NY = ZoneInfo('America/New_York')
TODAY = '2026-09-15'

def fetch(spec):
    symbol, interval, span = spec
    url = 'https://query2.finance.yahoo.com/v8/finance/chart/' + urllib.parse.quote(symbol) + f'?range={span}&interval={interval}&includePrePost=false&events=div%2Csplits'
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl.create_default_context(cafile=certifi.where()), timeout=20) as r:
            raw = json.load(r)
        (OUT / f'{symbol}-{interval}-raw.json').write_text(json.dumps(raw))
        result = raw['chart']['result'][0]
        q = result['indicators']['quote'][0]
        bars=[]
        for i,t in enumerate(result['timestamp']):
            row={k:q[k][i] for k in ('open','high','low','close','volume')}
            if any(v is None for v in row.values()): continue
            row['time']=datetime.fromtimestamp(t,NY).isoformat()
            bars.append(row)
        if interval=='1d':
            # Strict full-range gaps; report only remaining untraded interval.
            gaps=[]
            for i in range(1,len(bars)):
                prev,b=bars[i-1],bars[i]
                if b['low']>prev['high']:
                    lo,hi=prev['high'],min(x['low'] for x in bars[i:])
                    if hi>lo:gaps.append({'date':b['time'][:10],'type':'up','unfilled':[lo,hi]})
                elif b['high']<prev['low']:
                    lo,hi=max(x['high'] for x in bars[i:]),prev['low']
                    if hi>lo:gaps.append({'date':b['time'][:10],'type':'down','unfilled':[lo,hi]})
            return {'symbol':symbol,'interval':interval,'bars':bars[-3:],'unfilled_gaps_6mo':gaps,'source':url}
        today=[b for b in bars if b['time'][:10]==TODAY and '09:30'<=b['time'][11:16]<'16:00']
        prior=[b for b in bars if b['time'][:10]<TODAY and '09:30'<=b['time'][11:16]<'16:00']
        if not today or not prior:raise ValueError('missing today or previous session')
        cutoff=today[-1]['time'][11:16]
        vol=sum(b['volume'] for b in today)
        vwap=sum((b['high']+b['low']+b['close'])/3*b['volume'] for b in today)/vol if vol else None
        dates=sorted(set(b['time'][:10] for b in prior))
        pastvol=[sum(b['volume'] for b in prior if b['time'][:10]==d and b['time'][11:16]<=cutoff) for d in dates]
        orb=[b for b in today if b['time'][11:16]<'10:00']
        saved_quotes = {r['symbol']:r for r in json.loads((OUT/'quotes.json').read_text())}
        previous = saved_quotes[symbol]['previous_close']
        return {'symbol':symbol,'interval':interval,'last_bar':today[-1],'vwap_5m_approx':vwap,'opening_30m_high':max(b['high'] for b in orb),'opening_30m_low':min(b['low'] for b in orb),'open':today[0]['open'],'previous_close':previous,'open_gap_pct':(today[0]['open']/previous-1)*100,'relative_volume_4session_approx':vol/(sum(pastvol)/len(pastvol)) if all(pastvol) else None,'prior_sessions':len(dates),'source':url}
    except Exception as e:return {'symbol':symbol,'interval':interval,'error':str(e)}

if __name__=='__main__':
    specs=[(s,'5m','5d') for s in ['SPY','QQQ','SMH','XOM','CVX','NOW','AMD','MU','AVGO','ORCL','TSLA']]+[('TSLA','1d','6mo')]
    with ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(fetch,specs))
    out={'retrieved_at':datetime.now(NY).isoformat(),'method':'descriptive; typical-price 5m VWAP proxy; relative volume versus up to 4 prior sessions, includes partial current bar; raw unadjusted OHLC; strict full-range gaps over 6mo only','rows':rows}
    (OUT/'intraday.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps(out,ensure_ascii=False,indent=2))
