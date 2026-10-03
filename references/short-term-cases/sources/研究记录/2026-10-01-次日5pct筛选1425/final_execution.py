"""Fresh execution evidence, independently labelled from failed classifier direction."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('m',P/'model.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from src.analysis.short_term_factors import completed_regular_minutes
rows=pd.read_csv(P/'ranking-all.csv').set_index('symbol')
manifest=json.loads((P/'final/manifest.json').read_text());out=[]
for info in manifest:
 sym=info['symbol'];rawpath=P/'final/raw'/f'{sym}-live.json'
 assert info['ok'] and hashlib.sha256(rawpath.read_bytes()).hexdigest()==info['sha256']
 raw=json.loads(rawpath.read_text());meta=raw['chart']['result'][0]['meta'];q=completed_regular_minutes(raw,info['fetched_at']);d=m.daily(sym)
 clock=q.index[0].normalize()+pd.Timedelta(minutes=m.CLOCK)
 pre=q[q.index<clock];expected_min=m.CLOCK-570
 hist=m.minutes(sym);references=[]
 for day,g in hist.groupby(hist.index.date):
  if str(day)>=m.CFG['date']:continue
  b=g[g.index.hour*60+g.index.minute<m.CLOCK]
  if m.complete_reference_pre(b):references.append(float(b.volume.sum()))
 references=references[-20:]
 numerator_coverage=len(pre)/expected_min
 independent_rvol=float(pre.volume.sum()/np.mean(references)) if len(references)==20 and numerator_coverage>=.9 else None
 if sym in rows.index:fixed_rvol=float(rows.loc[sym,'rvol']);rvolsource='same fixed14:25 model signal'
 else:fixed_rvol=independent_rvol;rvolsource='independent minute-volume proxy; fixed model features incomplete, no classifier score'
 vols=float(q.volume.sum());vwap=float((((q.high+q.low+q.close)/3)*q.volume).sum()/vols)
 counts=q.close.resample('5min').count()
 b=q.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
 b=b[((b.index+pd.Timedelta(minutes=5))<=pd.Timestamp(info['fetched_at']).tz_convert(m.NY)) & (counts.reindex(b.index)==5)]
 # Each bar's VWAP reference includes actual completed minute data up to that bar.
 checks=[]
 for stamp,r in b.tail(2).iterrows():
  prefix=q[q.index<stamp+pd.Timedelta(minutes=5)];vv=prefix.volume.sum()
  barvw=float((((prefix.high+prefix.low+prefix.close)/3)*prefix.volume).sum()/vv)
  checks.append({'bar':stamp.isoformat(),'close':float(r.close),'vwap':barvw,'above':bool(r.close>barvw)})
 two=bool(len(checks)==2 and all(c['above'] for c in checks) and b.index[-1]-b.index[-2]==pd.Timedelta(minutes=5))
 df=m.daily_features(d,m.daily('QQQ')).iloc[-1];price=float(meta['regularMarketPrice'])
 stamp=pd.Timestamp(meta['regularMarketTime'],unit='s',tz='UTC').tz_convert(m.NY);age=(pd.Timestamp(info['fetched_at'])-stamp).total_seconds()
 eligible=bool(df.previous_close>=5 and df.adv20>=5e7)
 gate=bool(eligible and price>vwap and two and fixed_rvol is not None and fixed_rvol>=1.2 and age<=90)
 out.append({'symbol':sym,'price':price,'quote_at':stamp.isoformat(),'fetched_at':info['fetched_at'],'age_seconds_at_fetch':age,
  'change_pct':100*(price/df.previous_close-1),'vwap':vwap,'above_vwap':price>vwap,'last2_complete5m':checks,'two_consecutive_above':two,
  'fixed1425_rvol20':fixed_rvol,'rvol_source':rvolsource,'independent_minute_rvol20':independent_rvol,'reference_sessions':len(references),
  '1425_minute_coverage':numerator_coverage,'ATR14':float(df.atr),'ATR_pct':100*df.atr/price,'ma200':float(df.ma200),'RSI2_prior_daily':float(df.rsi2),
  'rsi14_prior_daily':None,'ret3_pct':100*df.ret3,'rs20_qqq_pp':100*df.rs20,'prior20_high':float(d.high.tail(20).max()),
  'eligible':eligible,'rightside_price_volume_gate':gate,'entry_cap':round(price+min(.005*price,.1*df.atr),2),
  'stop3_reference':round(price*.97,2),'target5_gross_reference':round(price*1.05,2),'target_net5_reference_cost002':round(price*1.052,2),
  'actual_bid_ask':None,'certified_probability':None,'model_direction_weight':0})
m.save('final/quotes-and-plan.json',out)
print(pd.DataFrame(out)[['symbol','price','quote_at','change_pct','vwap','fixed1425_rvol20','two_consecutive_above','rightside_price_volume_gate','entry_cap','stop3_reference','target5_gross_reference','target_net5_reference_cost002']].to_string(index=False))
