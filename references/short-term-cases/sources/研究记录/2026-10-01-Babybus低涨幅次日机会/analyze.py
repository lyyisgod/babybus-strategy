from pathlib import Path
import sys,json,shutil,hashlib
import pandas as pd,numpy as np
sys.path.insert(0,str(Path.cwd()))
from strategies.bb_washout_bounce.data import load_config,load_market,read_records,save,calendar
from strategies.bb_washout_bounce.backtest import Engine
from strategies.bb_washout_bounce.daily import daily_report
from strategies.bb_washout_bounce.factors import rsi,streak
P=Path(__file__).resolve().parent;cfg=load_config();protocol=json.loads((P/'protocol.json').read_text());secs=json.loads((P/'original-bb-universe.json').read_text());secs=secs.get('securities',secs) if isinstance(secs,dict) else secs
fresh=json.loads((P/'manifest-fresh.json').read_text());reused=json.loads((P/'manifest-reused.json').read_text());print('manifest-reused',str(reused)[:300],flush=True)
allman=reused+fresh if isinstance(reused,list) else reused['records']+fresh
lookup={(x['symbol'],x['mode']):x for x in allman}
D=P/'strict-raw';D.mkdir(exist_ok=True);manifest=[]
for s in protocol['stocks']+protocol['proxies']:
 a=lookup[(s,'daily')];src=P/'raw'/f'{s}-daily.json';raw=src.read_bytes();assert hashlib.sha256(raw).hexdigest()==a['sha256'];fn=s.replace('^','INDEX_').replace('=','_')+'.json';shutil.copy2(src,D/fn);manifest.append(dict(ticker=s,file=fn,source=a['url'],sha256=a['sha256'],fetched_at=a['fetched_at']))
save(D/'manifest.json',manifest)
market,metadata,errors,last=load_market(D,cfg,pd.Timestamp.now(tz='UTC'));market={s:d.loc['2021-01-01':] for s,d in market.items()};save(P/'loader-audit.json',dict(errors=errors,last=last,trim='2021-01-01: to satisfy original2020-2030calendar; originalbytes retained'))
records={k:read_records(Path('data/bb_washout_bounce')/(k+'.jsonl')) for k in ['fundamentals','macro','events','event_coverage','supports','conflicts']}
e=Engine(market,secs,records,cfg);res=daily_report(e,last,P/'strict-original',None,pd.Timestamp.now(tz='UTC'));print('STRICT',res,flush=True)
cal=calendar(cfg);sessions=cal.sessions_in_range('2021-01-01',cal.next_session(last));wend=pd.Series(sessions,index=sessions).groupby(sessions.to_period('W-FRI')).max();wend=pd.DatetimeIndex([d for d in wend if d<=last and cal.next_session(d).to_period('W-FRI')!=d.to_period('W-FRI')]);secmap={x['ticker']:x for x in secs}
cutmin=15*60+20;expected=pd.date_range('2026-10-01 09:30','2026-10-01 15:15',freq='5min').time

def intra(s,mode):
 a=json.loads((P/'raw'/f'{s}-{mode}.json').read_text())['chart']['result'][0];d=pd.DataFrame(a['indicators']['quote'][0],index=pd.to_datetime(a['timestamp'],unit='s',utc=True).tz_convert('America/New_York'));d=d[['open','high','low','close','volume']].dropna();d=d[(d.index.hour*60+d.index.minute>=570)&(d.index.hour*60+d.index.minute<960)];return a['meta'],d
rows=[]
for s in protocol['stocks']:
 try:
  d=market[s];price=d[['open','high','low','close']].mul(d.factor,axis=0);c=d.adjclose;meta,live=intra(s,'live');p=float(meta['regularMarketPrice']);prev=float(d.close.iloc[-1]);factor=float(d.factor.iloc[-1]);a=p*factor;weekly=c.reindex(wend).dropna();wm=float(weekly.tail(20).mean())/factor if len(weekly)>=20 else np.nan;ws=float(weekly.tail(20).std(ddof=0))/factor if len(weekly)>=20 else np.nan
  dd=1-a/price.high.tail(252).max();dd10=1-a/price.high.tail(10).max();low20=price.low.tail(20).min()/factor;low60=price.low.tail(60).min()/factor;rv90=np.log(c/c.shift()).tail(90).std(ddof=1)*np.sqrt(252);wv=weekly.pct_change(fill_method=None).tail(12).std(ddof=1) if len(weekly)>=13 else np.nan
  volref=float(d.volume.tail(20).mean());adv=float((d.close*d.volume).tail(20).mean());tr=pd.concat([price.high-price.low,(price.high-c.shift()).abs(),(price.low-c.shift()).abs()],axis=1).max(axis=1);atr=float(tr.tail(14).mean())/factor
  complete=live[live.index.minute%5==0].copy();groups=live.groupby(live.index.floor('5min'));b=groups.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'});count=groups['close'].count();b=b[count==5];b=b[(b.index.hour*60+b.index.minute<cutmin)]
  sub=live[live.index.hour*60+live.index.minute<cutmin];vwap=float((((sub.high+sub.low+sub.close)/3)*sub.volume).sum()/sub.volume.sum());todayvol=float(sub.volume.sum());two=bool(len(b)>=2 and (b.close.tail(2)>vwap).all());histmeta,hist=intra(s,'history');refs=[];refdates=[]
  for date,g in hist.groupby(hist.index.date):
   if str(date)>='2026-10-01':continue
   g=g[g.index.hour*60+g.index.minute<cutmin]
   if len(g)==len(expected) and list(g.index.time)==list(expected):refs.append(float(g.volume.sum()));refdates.append(str(date))
  rvol=todayvol/np.mean(refs[-20:]) if len(refs)>=20 and len(b)==len(expected) else np.nan
  f=e.factors[s].loc[last].to_dict() if s in e.factors else {}
  if not f:
   gr=streak(price.close>price.open);up=streak(c>c.shift());rs=streak(c.pct_change(fill_method=None)-market['SMH'].adjclose.pct_change(fill_method=None).reindex(c.index)>0);vr=d.volume/d.volume.shift().rolling(20).mean();candle=(price.close>price.open)&(vr>=1.5)&((price.close-price.low)/(price.high-price.low)>=.5)
   f=dict(green_streak=int(gr.iloc[-1]),up_close_streak=int(up.iloc[-1]),rs_lead_streak=int(rs.iloc[-1]),vol_ratio_20=float(vr.iloc[-1]),signal_bar=bool(candle.tail(2).any()))
  gates={'dd35_65':bool(.35<=dd<=.65),'rv90':bool(rv90>=.6),'weekly_vol':bool(wv>=.12),'washout_or_low20':bool(dd10>=.15 or 0<=p/low20-1<=.08),'bb_or_low60':bool(p<=wm or 0<=p/low60-1<=.12),'no_chase':bool(f.get('green_streak',3)<3 and f.get('up_close_streak',3)<3 and f.get('rs_lead_streak',3)<3),'prior_volume_green':bool(f.get('signal_bar',False) and f.get('vol_ratio_20',0)>=1.5 and f.get('green_streak',0)>=1)}
  row=dict(ticker=s,price=p,change_pct=(p/prev-1)*100,quote_at=pd.Timestamp(meta['regularMarketTime'],unit='s',tz='UTC'),fetched_at=lookup[(s,'live')]['fetched_at'],dd_high=dd,dd10=dd10,low20=low20,dist_low20=p/low20-1,low60=low60,dist_low60=p/low60-1,weekly_mid=wm,weekly_lower=wm-2*ws,rv90=rv90,weekly_vol=wv,adv_usd=adv,atr14=atr,ma200=float(c.tail(200).mean())/factor if len(c)>=200 else None,rsi2=float(rsi(c,2).iloc[-1]),rsi14=float(rsi(c,14).iloc[-1]),vwap1520=vwap,above_vwap=p>vwap,two_complete_above_vwap=two,rvol1520=rvol,rvol_reference_count=len(refs),rvol_reference_dates=refdates[-20:],full_5m_count=len(b),today_volume_fraction20=todayvol/volref,original_member=s in secmap,company_type=secmap.get(s,{}).get('company_type'),theme=secmap.get(s,{}).get('theme'),current_up_close_streak=(int(f.get('up_close_streak',0))+1 if p>prev else 0),gates=gates,gate_count=sum(gates.values()),prior_complete_factors=f,source=lookup[(s,'live')]['url'])
  rows.append(row)
 except Exception as err:print('ERROR',s,str(err),flush=True);rows.append(dict(ticker=s,error=str(err)))
save(P/'all-factors.json',rows);good=[r for r in rows if 'error' not in r and r['change_pct']<=3 and r['adv_usd']>=50e6];rank=sorted(good,key=lambda r:(r['company_type']=='infra_growth',r['gate_count'],r['above_vwap'] and r['two_complete_above_vwap'],r['rvol1520'] if np.isfinite(r['rvol1520']) else -1),reverse=True);save(P/'low-gain-ranked.json',rank)
for r in rank[:25]:print({k:r[k] for k in ['ticker','price','change_pct','dd_high','dd10','dist_low20','weekly_mid','rv90','weekly_vol','above_vwap','two_complete_above_vwap','rvol1520','gate_count','gates']},flush=True)
j=market['JNK'];w=j.adjclose.reindex(wend).dropna();monthly=j.adjclose.resample('ME').last();monthly=monthly.loc[monthly.index<=last];credit=dict(completed_session=last,weekly_end=w.index[-1],return20=float(j.adjclose.iloc[-1]/j.adjclose.iloc[-21]-1),weekly_break=bool(w.iloc[-1]<w.iloc[-9:-1].min()*.995),weekly_close=float(w.iloc[-1]),previous8low=float(w.iloc[-9:-1].min()),month_end=monthly.index[-1],ma3=float(monthly.tail(3).mean()),ma10=float(monthly.tail(10).mean()),bear_month=bool(monthly.tail(3).mean()<monthly.tail(10).mean()),oas20_missing=True,three_day_confirmation_not_computed=True);save(P/'credit-context.json',credit);print('CREDIT',credit,flush=True)
