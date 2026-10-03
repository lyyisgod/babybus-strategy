"""Weekend price-factor companion; not a certified Babybus strategy or probability."""
import sys,json,ssl,urllib.request,urllib.parse
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from strategies.bb_washout_bounce.data import load_config,load_market,save,clean
HERE=Path(__file__).resolve().parent
import certifi
CFG=load_config();NOW=datetime.now(timezone.utc)
market,meta,errors,last=load_market(HERE/'raw',CFG,NOW.isoformat())
NAMES=['CBRS','NBIS','IREN','APLD','ALAB','CRDO','LITE','VRT','OKLO','SMR']
# Fixed before opening outcomes. Core names, cyclicals and known CRWV dilution excluded.
FEATURES=['r3','r20','dd252','rv90','volume_ratio','close_range','rs20','sector5','credit20']
protocol=dict(created_at=NOW.isoformat(),names=NAMES,features=FEATURES,k=40,min_history=60,
 signal='Thursday completed close',entry='Following Friday official close proxy',exit='Immediately following calendar Monday close',
 outcomes={'strong_close':.05,'strong_high':.08,'loss_low':-.05},cost=.002,
 rank='Among live price above intraday VWAP, +0% to +6% today, >=SMH today, no known dilution, sort neighbor mean net return divided by negative P10 magnitude. Diagnostic ranking only; no parameter selection.',
 caveat='Current universe survivorship and repeated research; current-Friday filters not included in historical probability. Thursday-state model only. Strict Babybus eligibility reported separately.')
if not (HERE/'weekend_protocol.json').exists():save(HERE/'weekend_protocol.json',protocol)

def get_intraday(s):
 p=HERE/f'{s}-5m.json'
 if '--offline' in sys.argv:raw=json.loads(p.read_text())
 else:
  url='https://query2.finance.yahoo.com/v8/finance/chart/'+urllib.parse.quote(s,safe='')+'?range=1mo&interval=5m'
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
  with urllib.request.urlopen(req,timeout=30,context=ssl.create_default_context(cafile=certifi.where())) as r:body=json.load(r)
  raw={'source':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),'body':body};save(p,raw)
 a=raw['body']['chart']['result'][0];m=a['meta'];f=pd.DataFrame(a['indicators']['quote'][0],index=pd.to_datetime(a['timestamp'],unit='s',utc=True).tz_convert('America/New_York')).dropna(subset=['close'])
 reg=f.between_time('09:30','15:59');now=pd.Timestamp(raw['retrieved_at']).tz_convert('America/New_York')
 today=reg.loc[(reg.index.date==now.date()) & (reg.index+pd.Timedelta(minutes=5)<=now)]
 vwap=float(((today.high+today.low+today.close)/3*today.volume).sum()/today.volume.sum())
 end=today.index[-1].hour*60+today.index[-1].minute
 old=reg.loc[(reg.index.date<now.date()) & (reg.index.hour*60+reg.index.minute<=end)]
 totals=old.groupby(old.index.date).volume.sum().tail(20)
 prev=float(market[s].close.iloc[-1]);p=float(m['regularMarketPrice'])
 return s,dict(price=p,time=datetime.fromtimestamp(m['regularMarketTime'],timezone.utc).isoformat(),change=p/prev-1,prev=prev,vwap=vwap,relative_volume=float(today.volume.sum()/totals.median()),volume_days=len(totals),day_low=float(m['regularMarketDayLow']),day_high=float(m['regularMarketDayHigh']),open=float(today.open.iloc[0]))
with ThreadPoolExecutor(max_workers=4) as pool:live=dict(pool.map(get_intraday,NAMES+['SMH','JNK']))

def features(s):
 d=market[s];c=d.adjclose;hi=d.high*d.factor;lo=d.low*d.factor
 f=pd.DataFrame(index=d.index)
 f['r3']=c.pct_change(3);f['r20']=c.pct_change(20);f['dd252']=1-c/hi.rolling(252,min_periods=20).max()
 f['rv90']=np.log(c).diff().rolling(90).std()*np.sqrt(252)
 f['volume_ratio']=d.volume/d.volume.shift().rolling(20).mean()
 f['close_range']=(c-lo)/(hi-lo).replace(0,np.nan)
 f['rs20']=f.r20-market['SMH'].adjclose.pct_change(20)
 f['sector5']=market['SMH'].adjclose.pct_change(5)
 f['credit20']=market['JNK'].adjclose.pct_change(20)
 f['next_entry']=d.close.shift(-1);f['monday_close']=d.close.shift(-2)/d.close.shift(-1)-1
 f['monday_high']=d.high.shift(-2)/d.close.shift(-1)-1;f['monday_low']=d.low.shift(-2)/d.close.shift(-1)-1
 # Only Thu -> Fri -> Mon, not holidays leading to Tue; no cash dividends modeled.
 idx=d.index
 valid=[i for i in range(len(d)-2) if idx[i].weekday()==3 and idx[i+1].weekday()==4 and idx[i+2].weekday()==0 and (idx[i+2]-idx[i+1]).days==3]
 events=f.iloc[valid].dropna(subset=FEATURES+['monday_close','monday_high','monday_low']).copy()
 events['entry_date']=[str(idx[d.index.get_loc(t)+1].date()) for t in events.index]
 events['exit_date']=[str(idx[d.index.get_loc(t)+2].date()) for t in events.index]
 return f,events

def nearest(hist,cur):
 x=hist[FEATURES].to_numpy();med=np.median(x,axis=0);scale=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0);scale=np.where(scale>1e-8,scale,1)
 dist=np.mean(((x-cur[FEATURES].to_numpy(dtype=float))/scale)**2,axis=1)
 return hist.iloc[np.argsort(dist)[:40]]

rows=[]
for s in NAMES:
 f,ev=features(s);snap=live[s];cur=f.iloc[-1]
 row={'symbol':s,'live':snap,'features':{k:float(cur[k]) for k in FEATURES},'weekend_n':len(ev),'signal_date':str(last.date()),'eligible_live':bool(snap['price']>snap['vwap'] and 0<snap['change']<=.06 and snap['change']>=live['SMH']['change'])}
 if len(ev)>=60 and cur[FEATURES].notna().all():
  ns=nearest(ev,cur);ret=ns.monday_close-.002
  row['analog']={'n':len(ns),'up_count':int((ret>0).sum()),'close5_count':int((ns.monday_close>=.05).sum()),'touch8_count':int((ns.monday_high>=.08).sum()),'low_minus5_count':int((ns.monday_low<=-.05).sum()),'mean_net':float(ret.mean()),'median_net':float(ret.median()),'p10_net':float(ret.quantile(.1)),'p90_net':float(ret.quantile(.9)),'high_p75':float(ns.monday_high.quantile(.75))}
  row['rank_ratio']=row['analog']['mean_net']/max(.01,abs(row['analog']['p10_net']))
  ns.to_csv(HERE/f'{s}-analogs.csv',index_label='signal_date')
  oos=[]
  for i in range(max(60,len(ev)-80),len(ev)):
   history=ev.iloc[:i];current=ev.iloc[i];n=nearest(history,current)
   assert max(history.exit_date)<str(ev.index[i].date())
   oos.append({'signal':str(ev.index[i].date()),'forecast_up':float((n.monday_close>.002).mean()),'actual_up':bool(current.monday_close>.002),'baseline_up':float((history.monday_close>.002).mean()),'forecast_close5':float((n.monday_close>=.05).mean()),'actual_close5':bool(current.monday_close>=.05),'baseline_close5':float((history.monday_close>=.05).mean())})
  z=pd.DataFrame(oos);z.to_csv(HERE/f'{s}-oos.csv',index=False)
  out={}
  for label in ['up','close5']:
   brier=float(((z['forecast_'+label]-z['actual_'+label].astype(float))**2).mean());base=float(((z['baseline_'+label]-z['actual_'+label].astype(float))**2).mean())
   out[label]={'n':len(z),'brier':brier,'baseline_brier':base,'skill':1-brier/base if base else None}
  row['oos']=out
 else:row['model_status']='insufficient_same_business_weekend_history_or_features'
 # NBIS before restructured trading is not comparable.
 if s=='NBIS':row['business_history_warning']='Pre-2024 Yandex history would be incomparable; do not extrapolate across predecessor.'
 rows.append(row)
save(HERE/'weekend-results.json',dict(asof=NOW.isoformat(),rows=rows,live_context={s:live[s] for s in ['SMH','JNK']},errors=errors))
print(json.dumps(clean(rows),ensure_ascii=False,indent=2))
