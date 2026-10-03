from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import model as m
P=m.ROOT
cfg=json.loads((P/'coefficients.json').read_text());cols=cfg['features'];mu=np.array(cfg['train_mean']);sd=np.array(cfg['train_sd'])
ds={};errors=[]
for s in m.STOCKS+m.PROXIES:
 try: ds[s]=m.load(s)[0]
 except Exception as e: errors.append(dict(symbol=s,error=str(e)))
rows=[];live=[]
for s in m.STOCKS:
 if s not in ds:continue
 d=ds[s];f=m.features(d,ds).iloc[-1]
 if f.isna().any():errors.append(dict(symbol=s,error='incomplete features'));continue
 c=d.raw_close.iloc[-1];atr=f.atr_pct*c;adv=(d.raw_close*d.volume).shift().rolling(20).mean().iloc[-1]
 q=m.quote(s,d);t=q['last_at'];q['session']='overnight' if t.date()==m.DAY.date() and t.hour>=20 else 'after_hours' if t.date()==m.DAY.date() and t.hour>=16 else 'regular_or_stale'
 b=json.loads((P/'raw'/f'{s}-5m.json').read_text())['chart']['result'][0];bars=pd.DataFrame(b['indicators']['quote'][0],index=pd.to_datetime(b['timestamp'],unit='s',utc=True).tz_convert(m.NY));regular=bars[(bars.index>=m.DAY+pd.Timedelta(hours=9,minutes=30))&(bars.index<m.DAY+pd.Timedelta(hours=16))].dropna(subset=['close','volume']);vwap=(regular.close*regular.volume).sum()/regular.volume.sum()
 rows.append(dict(symbol=s,eligible=bool(c>=5 and adv>=50e6),close=c,daily_pct=f.ret1*100,volume_ratio=f.volume_ratio,atr=atr,atr_pct=f.atr_pct*100,rsi=f.rsi14*100,ema20=c/(1+f.ema20_dist),ema50=c/(1+f.ema50_dist),rs20_qqq_pp=f.rs20*100,ret20_pct=f.ret20*100,range_position=f.range_position,high=d.raw_high.iloc[-1],low=d.raw_low.iloc[-1],prior_high20=d.raw_high.iloc[-21:-1].max(),adv20=adv,history_sessions=len(d),regular_vwap_proxy=vwap,**{k:v for k,v in q.items() if k!='symbol'}))
 live.append(f)
x=pd.DataFrame(live);z=np.clip((x[cols].to_numpy()-mu)/sd,-8,8)
inter=[('ret1','volume_ratio'),('range_position','volume_ratio'),('ret1','QQQ_r1'),('atr_pct','VIX_level'),('ret20','rs20')]
a=pd.DataFrame(rows)
for kind in ['a','b']:
 design=np.c_[np.ones(len(z)),z] if kind=='a' else np.c_[np.ones(len(z)),z,z*z,*[z[:,cols.index(l)]*z[:,cols.index(r)] for l,r in inter]]
 for target in m.TARGETS:
  v=cfg['models'][f'{kind}_{target}'];beta=np.array(v['coefficients']);cb=v['calibration'];a[f'{kind}_{target}']=m.sigmoid(cb[0]+cb[1]*(design@beta))
for target in m.TARGETS:a[f'p_{target}']=(a[f'a_{target}']+a[f'b_{target}'])/2
old=pd.read_csv(P.parent/'ranking.csv');a['extension_atr']=(a.close-a.ema20)/a.atr
a=a.merge(old[['symbol','last','p_hit5']].rename(columns={'last':'previous_last','p_hit5':'previous_p_hit5'}),on='symbol');a['price_change_from_previous_pct']=(a['last']/a.previous_last-1)*100;a['hit5_change_pp']=(a.p_hit5-a.previous_p_hit5)*100
a=a.sort_values('p_hit5',ascending=False);a.to_csv(P/'ranking.csv',index=False);a[a.eligible].to_csv(P/'eligible-ranking.csv',index=False);m.save('ranking.json',a.to_dict('records'));m.save('errors.json',errors);m.save('data-reconciliation.json',m.RECONCILED);m.save('daily-field-fallbacks.json',m.FALLBACKS)
x.assign(symbol=[r['symbol'] for r in rows]).to_csv(P/'latest-features.csv',index=False)
manifest=json.loads((P/'manifest.json').read_text());print(type(manifest),len(manifest))
assert np.isfinite(a[[f'p_{t}' for t in m.TARGETS]].to_numpy()).all();assert (a.p_hit10<=a.p_hit5+.01).all();assert all(ds[s].index.max()==m.DAY for s in a.symbol)
for s in ['ALAB','NBIS','CRDO','COHR','AAOI','BA']:
 d=ds[s];cut=300;pd.testing.assert_frame_equal(m.features(d,ds).iloc[:cut],m.features(d.iloc[:cut],ds))
assert hashlib.sha256((P/'coefficients.json').read_bytes()).hexdigest()==m.P['coefficients_sha256']
m.save('validation.json',dict(status='pass',frozen_coefficient_sha256=m.P['coefficients_sha256'],loaded=len(a),eligible=int(a.eligible.sum()),errors=errors,checks=['all stock daily features end Sep29','feature prefix causality for six selected/event names','frozen model fingerprint unchanged','finite calibrated ensemble scores','nested targets'],not_validated=['live overnight executable quotes','next-day event outcome','current-night entry probability']))
print(a[a.symbol.isin(['ALAB','NBIS','CRDO','COHR','AAOI','BA','IOVA','KOD','CBRS'])][['symbol','close','last','last_at','session','price_change_from_previous_pct','p_hit5','p_hit10','p_down3','hit5_change_pp','volume_ratio','atr','high','low','regular_vwap_proxy']].to_string(index=False))
