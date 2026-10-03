import json,pandas as pd,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;rows={x['ticker']:x for x in json.loads((p/'all-factors.json').read_text())};output=[]
for m in json.loads((p/'publication/manifest.json').read_text()):
 s=m['symbol'];a=json.loads((p/'publication/raw'/f'{s}-live.json').read_text())['chart']['result'][0];meta=a['meta'];d=pd.DataFrame(a['indicators']['quote'][0],index=pd.to_datetime(a['timestamp'],unit='s',utc=True).tz_convert('America/New_York'));d=d[(d.index.hour*60+d.index.minute>=570)&(d.index.hour*60+d.index.minute<960)].dropna();at=pd.Timestamp(meta['regularMarketTime'],unit='s',tz='UTC').tz_convert('America/New_York');cut=at.floor('5min');sub=d[d.index<cut];groups=sub.groupby(sub.index.floor('5min'));b=groups.agg({'open':'first','close':'last','volume':'sum'});b=b[groups.close.count()==5];vwap=float(((sub.high+sub.low+sub.close)/3*sub.volume).sum()/sub.volume.sum());r=rows[s];px=float(meta['regularMarketPrice']);prev=meta['previousClose'];h=json.loads((p/'raw'/f'{s}-history.json').read_text())['chart']['result'][0];hist=pd.DataFrame(h['indicators']['quote'][0],index=pd.to_datetime(h['timestamp'],unit='s',utc=True).tz_convert('America/New_York')).dropna();refs=[];expected=pd.date_range(pd.Timestamp('2026-10-01 09:30',tz='America/New_York'),cut-pd.Timedelta(minutes=5),freq='5min').time
 for date,g in hist.groupby(hist.index.date):
  if str(date)>='2026-10-01':continue
  mins=g.index.hour*60+g.index.minute;g=g[(mins>=570)&(mins<cut.hour*60+cut.minute)]
  if len(g)==len(expected) and list(g.index.time)==list(expected):refs.append(float(g.volume.sum()))
 rvol=float(sub.volume.sum()/np.mean(refs[-20:])) if len(refs)>=20 and len(b)==len(expected) else None
 o=dict(ticker=s,price=px,change_pct=(px/prev-1)*100,quote_at=at.isoformat(),fetched_at=m['fetched_at'],source=m['url'],prior_close=prev,day_open=float(sub.open.iloc[0]),day_high=meta['regularMarketDayHigh'],day_low=meta['regularMarketDayLow'],vwap=vwap,bar_cutoff=cut.isoformat(),rvol_sameclock=rvol,rvol_reference_count=len(refs),complete5m_count=len(b),expected5m_count=len(expected),two_closes_above_vwap=bool((b.close.tail(2)>vwap).all()),above_vwap=px>vwap,green_vs_open=px>float(sub.open.iloc[0]),cap=px+min(.005*px,.1*r['atr14']),stop3=px*.97,target5=px*1.05,target_net5_proxy=px*1.052,atr14=r['atr14'],today_upclosestreak=(r['prior_complete_factors'].get('up_close_streak',0)+1 if px>prev else 0),probability_tomorrow5=None,orders_sent=False)
 output.append(o);print(o)
(p/'publication/quotes-and-plan.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print('ANNUAL IREN',707/501-1,'QUARTER IREN',137.2/187.3-1,'APLD',258748/51076-1)
