"""Fixed-universe descriptive event study. Not a calibrated forecast."""
import json, math, ssl, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import certifi
import pandas as pd
R=Path(__file__).resolve().parent
intraday=json.loads((R/'results.json').read_text())
SYMBOLS=intraday['hardware_universe_fixed_before_fetch']+['SMH','SOXX','QQQ']
def get(s):
 u='https://query1.finance.yahoo.com/v8/finance/chart/'+urllib.parse.quote(s,safe='')+'?range=5y&interval=1d&events=div%2Csplits'
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=15,context=ssl.create_default_context(cafile=certifi.where())) as r:raw=json.load(r)
  (R/(s+'-daily.json')).write_text(json.dumps(raw))
  x=raw['chart']['result'][0];q=x['indicators']['quote'][0]
  d=pd.DataFrame(q,index=pd.to_datetime(x['timestamp'],unit='s',utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d'))
  adj=x['indicators'].get('adjclose',[{}])[0].get('adjclose')
  if adj is None:raise ValueError('adjusted close unavailable')
  d['adjclose']=adj;d=d[d.index<'2026-09-14'].dropna()
  factor=d.adjclose/d.close
  for k in ['open','high','low','close']:d['a_'+k]=d[k]*factor
  return s,d
 except Exception as e:return s,{'error':str(e)}
with ThreadPoolExecutor(max_workers=6) as ex:frames=dict(ex.map(get,SYMBOLS))
def wilson(k,n):
 if not n:return None
 z=1.95996398454;p=k/n;den=1+z*z/n
 a=(p+z*z/(2*n))/den;b=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [a-b,a+b]
assert abs(wilson(5,10)[0]-(1-wilson(5,10)[1]))<1e-9
smh=frames['SMH'].a_close.pct_change()
out={'retrieved_at':datetime.now(timezone.utc).isoformat(),'universe':SYMBOLS,'definition':'Historical completed day: stock adjusted return <= -2%, SMH <= -2%, stock close location >=0.60; next-day adjusted open entry; exit close on 3rd/5th holding session; 0.2% roundtrip cost; greedy nonoverlapping holding windows; current intraday screen is provisional, not equivalent to finalized signal.','by_stock':{}}
for s,d in frames.items():
 if isinstance(d,dict):out['by_stock'][s]=d;continue
 ret=d.a_close.pct_change();clv=(d.close-d.low)/(d.high-d.low)
 mask=(ret<=-.02)&(smh.reindex(d.index)<=-.02)&(clv>=.60)
 entryids=[i for i in range(len(d)) if mask.iloc[i]]
 stats={}
 for h in [3,5]:
  ids=[];last=-999
  for i in entryids:
   if i+h<len(d) and i>last+h:ids.append(i);last=i
  rets=[d.a_close.iloc[i+h]/d.a_open.iloc[i+1]-1-.002 for i in ids]
  adverse=[min(d.a_low.iloc[i+1:i+h+1])/d.a_open.iloc[i+1]-1 for i in ids]
  wins=sum(v>0 for v in rets)
  stats[str(h)]={'n':len(ids),'win_frequency':wins/len(ids) if ids else None,'wilson95':wilson(wins,len(ids)),'median_net_return':float(pd.Series(rets,dtype=float).median()) if ids else None,'median_adverse_excursion':float(pd.Series(adverse,dtype=float).median()) if ids else None,'event_dates':[d.index[i] for i in ids]}
 c=d.close;tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
 out['by_stock'][s]={'events':stats,'prior_close':float(c.iloc[-1]),'ma20':float(c.rolling(20).mean().iloc[-1]),'ma50':float(c.rolling(50).mean().iloc[-1]),'atr14':float(tr.rolling(14).mean().iloc[-1]),'prior20_low':float(d.low.tail(20).min()),'prior20_return_pct':float((c.iloc[-1]/c.iloc[-21]-1)*100)}
(R/'event-results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for s in ['NVDA','AMD','MU','TSM','AVGO','MRVL','AMAT','TXN']:
 print(s,json.dumps({k:({h:{kk:vv for kk,vv in a.items() if kk!='event_dates'} for h,a in v.items()} if k=='events' else v) for k,v in out['by_stock'][s].items()},ensure_ascii=False))
