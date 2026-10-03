from pathlib import Path
import sys,json,math,hashlib
import pandas as pd,numpy as np
sys.path.insert(0,str(Path.cwd()))
from strategies.bb_washout_bounce.data import load_config,calendar,save
from strategies.bb_washout_bounce.factors import rsi
P=Path(__file__).resolve().parent;proto=json.loads((P/'protocol.json').read_text());cal=calendar(load_config());NY='America/New_York';output={};events=[];rejects=[]
def read(s,mode):
 a=json.loads((P/'raw'/f'{s}-{mode}.json').read_text())['chart']['result'][0];d=pd.DataFrame(a['indicators']['quote'][0],index=pd.to_datetime(a['timestamp'],unit='s',utc=True).tz_convert(NY));return a,d

def wilson(k,n):
 if n==0:return None
 z=1.959963984540054;p=k/n;den=1+z*z/n;ctr=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [max(0,ctr-half),min(1,ctr+half)]

def stats(v):
 fields=['hit_up5','close_net5','hit_down5','hit_down3','break_known_intradaylow','close_negative','break_prior20low','firstwin5_before_stop3'];res={'n':len(v)}
 for name in fields:
  known=[r[name] for r in v if r[name] is not None];k=sum(known);n=len(known);res[name]={'count':k,'n':n,'frequency':k/n if n else None,'wilson95':wilson(k,n)}
 if v:res.update(mean_net_return=float(np.mean([r['close_net_return'] for r in v])),median_net_return=float(np.median([r['close_net_return'] for r in v])),empirical_return_p10_p90=np.quantile([r['close_net_return'] for r in v],[.1,.9]).tolist())
 return res

for s in proto['symbols']:
 a,h=read(s,'history');mins=h.index.hour*60+h.index.minute;h=h[(mins>=570)&(mins<960)].dropna(subset=['open','high','low','close','volume']);groups={str(day):g for day,g in h.groupby(h.index.date)};days=cal.sessions_in_range(min(groups),'2026-09-30');dres,d=read(s,'daily');d.index=d.index.tz_localize(None).normalize();d=d.loc[:'2026-09-30'];factor=np.array(dres['indicators'].get('adjclose',[{'adjclose':d.close.tolist()}])[0]['adjclose'])[:len(d)]/d.close.to_numpy();d['adjclose']=d.close*np.array(factor);d['factor']=factor;adj=d[['open','high','low','close']].mul(d.factor,axis=0);r14=rsi(d.adjclose,14);asset=[]
 expected=list(pd.date_range('2000-01-01 09:30','2000-01-01 15:55',freq='5min').time)
 for position,session in enumerate(days):
  nxt=cal.next_session(session);date=str(session.date());nxtdate=str(nxt.date());g=groups.get(date);ng=groups.get(nxtdate)
  if nxtdate>='2026-10-01':rejects.append(dict(ticker=s,date=date,reason='unmatured nextday'));continue
  if g is None or ng is None or len(g)!=78 or len(ng)!=78 or list(g.index.time)!=expected or list(ng.index.time)!=expected:rejects.append(dict(ticker=s,date=date,reason='missing/notexact78regular5mbars'));continue
  prior=d.loc[d.index<session];known=g[g.index.hour*60+g.index.minute<930];futuretoday=g[g.index.hour*60+g.index.minute>=930];tom=ng[ng.index.hour*60+ng.index.minute<955];assert len(known)==72 and len(tom)==77
  entry=float(known.close.iloc[-1]);high=float(tom.high.max());low=float(tom.low.min());exitpx=float(tom.close.iloc[-1]);knownlow=float(known.low.min());previouslow=float((prior.low*prior.factor).tail(20).min()/prior.factor.iloc[-1]) if len(prior)>=20 else None;ma200=float(prior.adjclose.tail(200).mean()/prior.factor.iloc[-1]) if len(prior)>=200 else None;rs=float(r14.loc[prior.index[-1]]) if len(prior) else None;prevclose=float(prior.close.iloc[-1]);sim=entry<=prevclose and ma200 is not None and entry<ma200 and rs is not None and rs<50
  firstwin=False;path=pd.concat([futuretoday,tom]);tp=entry*1.05;stop=entry*.97
  for _,b in path.iterrows():
   if b.open<=stop:break
   if b.open>=tp:firstwin=True;break
   if b.low<=stop:break
   if b.high>=tp:firstwin=True;break
  r=dict(ticker=s,date=date,next_session=nxtdate,calendar_position=position,nonoverlap_even=position%2==0,entry=entry,exit1555=exitpx,high_tom=high,low_tom=low,known_intradaylow=knownlow,prior20low=previouslow,prior_rsi14=rs,prior_ma200=ma200,similar_state=sim,hit_up5=high/entry-1>=.05,close_net5=exitpx/entry-1-.002>=.05,hit_down5=low/entry-1<=-.05,hit_down3=low/entry-1<=-.03,break_known_intradaylow=low<knownlow,close_negative=exitpx<entry,break_prior20low=(low<previouslow if previouslow is not None else None),firstwin5_before_stop3=firstwin,close_net_return=exitpx/entry-1-.002,high_return=high/entry-1,low_return=low/entry-1)
  asset.append(r);events.append(r)
 base=stats([r for r in asset if r['nonoverlap_even']]);odd=stats([r for r in asset if not r['nonoverlap_even']]);sim=stats([r for r in asset if r['nonoverlap_even'] and r['similar_state']]);sens={}
 for shift in [-.0025,.0025]:
  v=[]
  for r in asset:
   if not r['nonoverlap_even']:continue
   ent=r['entry']*(1+shift);v.append({'hit_up5':r['high_tom']/ent-1>=.05,'close_net5':r['exit1555']/ent-1-.002>=.05,'hit_down5':r['low_tom']/ent-1<=-.05,'hit_down3':r['low_tom']/ent-1<=-.03,'break_known_intradaylow':r['break_known_intradaylow'],'close_negative':r['exit1555']<ent,'break_prior20low':r['break_prior20low'],'firstwin5_before_stop3':None,'close_net_return':r['exit1555']/ent-1-.002})
  sens[str(shift)]=stats(v)
 lmeta,live=read(s,'live');qt=pd.Timestamp(lmeta['meta']['regularMarketTime'],unit='s',tz='UTC').tz_convert(NY);m=live.index.hour*60+live.index.minute;live=live[(m>=570)&(m<960)&(live.index<qt.floor('min'))].dropna();px=float(lmeta['meta']['regularMarketPrice']);vw=float((((live.high+live.low+live.close)/3)*live.volume).sum()/live.volume.sum());cut=live[live.index.hour*60+live.index.minute<930];refpx=float(cut.close.iloc[-1]);vol90=float(np.log(d.adjclose/d.adjclose.shift()).tail(90).std(ddof=1)*math.sqrt(252));daily_sigma=vol90/math.sqrt(252);cdf=lambda x:.5*(1+math.erf(x/math.sqrt(2)))
 current=dict(price=px,change_pct=(px/lmeta['meta']['previousClose']-1)*100,quote_at=qt,price1530=refpx,latest_vs1530=px/refpx-1,vwap=vw,day_low=lmeta['meta']['regularMarketDayLow'],day_high=lmeta['meta']['regularMarketDayHigh'],source=lmeta['meta']['symbol'],gaussian_zero_drift_daily_close5_up=1-cdf(math.log(1.05)/daily_sigma),gaussian_zero_drift_daily_close5_down=cdf(math.log(.95)/daily_sigma),gaussian_not_direction_forecast=True)
 output[s]=dict(history_start=asset[0]['date'] if asset else None,history_end=asset[-1]['date'] if asset else None,all_complete_diagnostic=stats(asset),primary_nonoverlap=base,opposite_parity_sensitivity=odd,similar_state_diagnostic=sim,entry_sensitivity=sens,current=current,calibrated_current_probability=None,model_abstains_for_direction=True)
 print(s,json.dumps({'primary':base,'opposite_parity':odd,'similar':sim,'current':{k:str(v) for k,v in current.items()}},ensure_ascii=False),flush=True)
save(P/'results.json',output);save(P/'historical-scenarios.json',events);save(P/'exclusions.json',rejects)
# Numerical and maturitychecks independent of investment validity.
assert abs(wilson(0,10)[1]-.2775327998628892)<1e-10 and wilson(0,0) is None
assert len(expected)==78
for s in proto['symbols']:
 v=[x for x in events if x['ticker']==s and x['nonoverlap_even']];assert all(pd.Timestamp(v[i]['next_session'])<pd.Timestamp(v[i+1]['date']) for i in range(len(v)-1));assert all(x['next_session']<'2026-10-01' for x in v);assert all(not x['close_net5'] or x['hit_up5'] for x in v)
for m in json.loads((P/'manifest-reused.json').read_text())+json.loads((P/'manifest-live.json').read_text()):assert hashlib.sha256((P/'raw'/f"{m['symbol']}-{m['mode']}.json").read_bytes()).hexdigest()==m['sha256']
save(P/'verification.json',dict(wilson_boundary=True,nonoverlap=True,labels_mature=True,fullregular78bars=True,raw_hashes=True,investment_probability_validated=False,conditional_Babybus_probability=None))
