import json,math,statistics,re
from datetime import datetime,date
from zoneinfo import ZoneInfo
from pathlib import Path
R=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
def chart(s,k='1d'):return json.loads((R/'raw'/f'{s}-{k}.json').read_text())['chart']['result'][0]
def bars(s):
 d=chart(s);q=d['indicators']['quote'][0];a=d['indicators'].get('adjclose',[{}])[0].get('adjclose',q['close'])
 return [dict(date=datetime.fromtimestamp(t,NY).date().isoformat(),**{k:q[k][i] for k in q},adj=a[i]) for i,t in enumerate(d['timestamp']) if q['close'][i] is not None]
spy=bars('SPY');spy20=spy[-1]['adj']/spy[-21]['adj']-1
out={}
for s in ['AVAV','KTOS','RCAT','ONDS']:
 b=bars(s);c=[x['close'] for x in b];a=[x['adj'] for x in b];lr=[math.log(a[i]/a[i-1]) for i in range(1,len(a))];rv=statistics.stdev(lr[-20:])*math.sqrt(252)
 tr=[max(b[i]['high']-b[i]['low'],abs(b[i]['high']-c[i-1]),abs(b[i]['low']-c[i-1])) for i in range(1,len(b))];atr=statistics.mean(tr[:14])
 for v in tr[14:]:atr=(atr*13+v)/14
 m=chart(s)['meta'];d=json.loads((R/'raw'/f'{s}-options.json').read_text());od=d['data'];opts=[]
 for o in od['options']:
  hit=re.search(r'(\d{6})([CP])(\d{8})$',o['option'])
  if not hit:continue
  exp=datetime.strptime(hit[1],'%y%m%d').date();k=int(hit[3])/1000
  if hit[2]=='C' and str(exp) in ['2026-09-18','2026-10-16','2026-11-20'] and c[-1]*.98<=k<=c[-1]*1.13:
   bid=o['bid'];ask=o['ask'];mid=(bid+ask)/2
   opts.append(dict(id=o['option'],expiry=str(exp),strike=k,bid=bid,ask=ask,spread_pct=100*(ask-bid)/mid if mid else None,iv=o['iv'],delta=o['delta'],theta=o['theta'],volume=o['volume'],oi=o['open_interest'],last_trade_time=o['last_trade_time'],breakeven=k+ask,breakeven_up_pct=((k+ask)/c[-1]-1)*100,double_up_pct=((k+2*ask)/c[-1]-1)*100))
 d5=chart(s,'5m');q=d5['indicators']['quote'][0];latest=[dict(time=datetime.fromtimestamp(t,NY).isoformat(),close=q['close'][i]) for i,t in enumerate(d5['timestamp']) if q['close'][i] is not None][-1]
 out[s]=dict(date=b[-1]['date'],close=c[-1],day_pct=100*(c[-1]/c[-2]-1),open=b[-1]['open'],high=b[-1]['high'],low=b[-1]['low'],post=latest,post_pct=100*(latest['close']/c[-1]-1),r5=100*(a[-1]/a[-6]-1),r20=100*(a[-1]/a[-21]-1),rs20=100*(a[-1]/a[-21]-1-spy20),ma20=statistics.mean(c[-20:]),ma50=statistics.mean(c[-50:]),prior20high=max(x['high'] for x in b[-21:-1]),low20=min(x['low'] for x in b[-20:]),volume_ratio=b[-1]['volume']/statistics.mean(x['volume'] for x in b[-21:-1]),rv20_pct=rv*100,atr14=atr,iv30_pct=od['iv30'],iv_rv=od['iv30']/100/rv,option_snapshot_raw_timestamp=d['timestamp'],options=opts,range5=[min(x['low'] for x in b[-5:]),max(x['high'] for x in b[-5:])])
(R/'model-results.json').write_text(json.dumps(out,indent=2))
for s,d in out.items():print(s,json.dumps(d,ensure_ascii=False))
