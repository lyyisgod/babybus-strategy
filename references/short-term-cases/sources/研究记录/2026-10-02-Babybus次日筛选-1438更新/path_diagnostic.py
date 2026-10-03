from pathlib import Path
import importlib.util,json,sys,hashlib
import pandas as pd,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(ROOT))
from strategies.bb_washout_bounce.data import save,load_config,load_market,calendar,clean
spec=importlib.util.spec_from_file_location('prior',ROOT/'研究记录/2026-10-01-夜盘Babybus全面筛选/scan.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cfg=load_config();now=pd.Timestamp.now(tz='UTC');market,_,_,last=load_market(P/'daily-raw',cfg,now.isoformat());cal=calendar(cfg)
proto=json.loads((P/'shortlist-protocol.json').read_text());stocks=[s for s in proto['names'] if s not in ['QQQ','SMH','JNK','SPY']]
live=[];events=[];baseline={};prefix={};parsed={}
manifest_by={r['ticker']:r for r in json.loads((P/'shortlist-raw/manifest.json').read_text())}
for s in proto['names']:
 snapshot_at=min(now,pd.Timestamp(manifest_by[s]['fetched_at']))
 a=json.loads((P/'shortlist-raw'/f'{s}.json').read_text())['chart']['result'][0];df=pd.DataFrame(a['indicators']['quote'][0],index=pd.to_datetime(a['timestamp'],unit='s',utc=True).tz_convert('America/New_York'))
 df=df.dropna(subset=['open','high','low','close','volume']);df=df[(df.index.hour*60+df.index.minute>=570)&(df.index.hour*60+df.index.minute<960)&(df.index+pd.Timedelta(minutes=5)<=snapshot_at)]
 parsed[s]=df;mins=df.index.hour*60+df.index.minute;cutoff=min(875,int(snapshot_at.tz_convert('America/New_York').hour*60+snapshot_at.tz_convert('America/New_York').minute)//5*5)
 groups={str(day):g for day,g in df.groupby(df.index.date)};cur=groups.get('2026-10-02',pd.DataFrame());past=[]
 expected=list(range(570,cutoff,5))
 for day,g in groups.items():
  gm=g.index.hour*60+g.index.minute;v=g[gm<cutoff]
  if day<'2026-10-02' and (v.index.hour*60+v.index.minute).tolist()==expected and len(g)==78:past.append((day,float(v.volume.sum())))
 gm=cur.index.hour*60+cur.index.minute;c=cur[gm<cutoff]
 volume=float(c.volume.sum());reference=past[-20:]
 vwap=float((((c.high+c.low+c.close)/3)*c.volume).sum()/volume) if volume else None
 mt=a['meta'];qt=pd.Timestamp(mt['regularMarketTime'],unit='s',tz='UTC').tz_convert('America/New_York');price=mt['regularMarketPrice'];prev=market[s].close.iloc[-1]
 live.append(dict(ticker=s,price=price,quote_at=qt,change_pct=(price/prev-1)*100,vwap_completed=vwap,cutoff_et=f'{cutoff//60:02}:{cutoff%60:02}',current_volume=volume,rvol20_same_clock=volume/np.mean([v for _,v in reference]) if len(reference)==20 else None,reference_days=len(reference),reference_dates=[d for d,_ in reference],above_vwap=price>vwap if vwap else None,low_completed=float(c.low.min()),high_completed=float(c.high.max()),clock14_35_pending=cutoff<875,bid_ask=None,source=f'https://finance.yahoo.com/quote/{s}/'))
 if s not in stocks:continue
 f=m.light(market[s],market['SMH'],cal);skip_until=None
 for day,g in groups.items():
  date=pd.Timestamp(day)
  if date>=last or len(g)!=78 or not cal.is_session(date):continue
  nxt=cal.next_session(date);ng=groups.get(str(nxt.date()))
  if ng is None or len(ng)!=78 or nxt>last:continue
  prior=cal.previous_session(date)
  if prior not in f.index:continue
  row=f.loc[prior];parts=m.gates(row);count=sum(parts.values());clock=g[g.index.hour*60+g.index.minute<875]
  if len(clock)!=61:continue
  p=float(clock.close.iloc[-1]);vw=float((((clock.high+clock.low+clock.close)/3)*clock.volume).sum()/clock.volume.sum())
  entrypath=g[g.index.hour*60+g.index.minute>=875];exitpath=ng[ng.index.hour*60+ng.index.minute<955];path=pd.concat([entrypath,exitpath]);stop=p*.97;target=p*1.05;exitp=float(exitpath.close.iloc[-1]);reason='time15:55'
  for ts,b in path.iterrows():
   if b.open<=stop:exitp=float(b.open);reason='gap_stop';break
   if b.open>=target:exitp=target;reason='gap_target_capped';break
   if b.low<=stop:exitp=stop;reason='stop_or_dualhit';break
   if b.high>=target:exitp=target;reason='target';break
  net=exitp/p-1-.002;baseline.setdefault(day,[]).append(net)
  if skip_until is not None and date<=skip_until:continue
  if count>=5 and row.rv90>=.6 and row.adv20>=50e6 and p>vw:
   skip_until=nxt
   events.append(dict(ticker=s,signal_session=str(prior.date()),entry_session=day,exit_session=str(nxt.date()),entry_clock='14:35',entry_price=p,daily_gates=count,net_return=net,exit_reason=reason,nextday_high_touch5=float(ng.high.max()/p-1)>=.05,nextday_high_touch10=float(ng.high.max()/p-1)>=.10,nextday_low_return=float(ng.low.min()/p-1),cost_double_net=net-.002,actual_path_not_original_babybus=True))
 for cut in [market[s].index[-25]]:
  t=m.light(market[s].loc[:cut],market['SMH'].loc[:cut],cal);cols=['dd_high','dd10','rv90','weekly_mid','weekly_vol','ma200','rsi2','atr14','vol_ratio20'];prefix[s]=bool(np.allclose(t.loc[cut,cols].astype(float),f.loc[cut,cols].astype(float),equal_nan=True))
stats=[];rng=np.random.default_rng(1002);dates=sorted(baseline)
for s in stocks:
 es=[e for e in events if e['ticker']==s];n=len(es)
 if not n:stats.append(dict(ticker=s,n=0,current_probability=None));continue
 a=np.zeros((len(dates),3))
 for e in es:
  i=dates.index(e['entry_session']);a[i]=[1,e['net_return'],e['net_return']-np.mean(baseline[e['entry_session']])]
 draws=[]
 for _ in range(1000):
  starts=rng.integers(0,max(1,len(a)-4),size=int(np.ceil(len(a)/5)));idx=np.concatenate([np.arange(i,min(i+5,len(a))) for i in starts])[:len(a)];tot=a[idx].sum(axis=0)
  if tot[0]:draws.append(tot[1:]/tot[0])
 stats.append(dict(ticker=s,n=n,wins=sum(e['net_return']>0 for e in es),touch5=sum(e['nextday_high_touch5'] for e in es),touch10=sum(e['nextday_high_touch10'] for e in es),mean_net=np.mean([e['net_return'] for e in es]),mean_net_dateblock95=np.quantile(np.array(draws)[:,0],[.025,.975]),excess_dateblock95=np.quantile(np.array(draws)[:,1],[.025,.975]),mean_doublecost=np.mean([e['cost_double_net'] for e in es]),current_probability=None,limitations='small samples; current survivor twelve-stock pool; repeatedly used period; signals ignore fundamentals/credit/event gates; historical14:35 proxy is not current conditional pullback entry'))
stats=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()} for r in stats]
mf=json.loads((P/'shortlist-raw/manifest.json').read_text());assert all(hashlib.sha256((P/'shortlist-raw'/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in mf if 'file' in r);assert all(prefix.values());assert all(e['exit_session']<='2026-10-01' for e in events)
save(P/'shortlist-live.json',live);save(P/'path-events.json',events);save(P/'path-stats.json',stats);save(P/'path-verification.json',dict(prefix=prefix,hashes=True,not_mature_today_excluded=True,nonoverlap_per_ticker=True,cost_roundtrip=.002,current_probability_certified=False))
for r in live:print(json.dumps(clean(r),ensure_ascii=False))
print('HISTORICAL',json.dumps(clean(stats),ensure_ascii=False))
