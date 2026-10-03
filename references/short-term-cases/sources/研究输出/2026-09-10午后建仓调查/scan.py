import json,datetime,urllib.request,ssl,certifi,concurrent.futures,sys
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent;NY=ZoneInfo('America/New_York');now=datetime.datetime.now(NY)
stocks=['VST','CEG','TLN','NRG','VRT','BE','OKLO','PWR','MU','SNDK','NVDA','AMD','AVGO','ARM','ANET','COHR','LITE','CRWV','NBIS','IREN','PLTR','ORCL','ALAB','CRDO','APP','HOOD','RKLB','ASTS','CRWD','NET']
stocks += ['XOM','CVX','COP','SLB','OXY','GE','GEV','ETN','CAT','LLY','JNJ','UNH','COST','WMT','AMZN','META','MSFT','GOOGL','AAPL','DDOG','PANW','UBER','JPM','GS']
bench=['QQQ','SMH','SPY','XLE','XLU','XLV','XLF','XLI','XLP','XLY','XLK','XLC','XLB','XLRE','IWM'];symbols=stocks+bench
config={'as_of':now.isoformat(),'horizon':'2-5 trading sessions','universe':stocks,'benchmarks':bench,'factors':'VWAP position, session recovery, QQQ-relative return, 20-day trend and relative momentum, RV20 and ATR14, 60-day market beta','method':'Unvalidated screening rules; no probability forecast; universe selected before this scan; all are proxies outside the project full-factor model.'}
(ROOT/'配置.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
def fetch(job):
 s,kind=job;u=f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range={"2y" if kind=="daily" else "1d"}&interval={"1d" if kind=="daily" else "1m"}&includePrePost=false&events=div%2Csplits'
 try:
  if '--saved' in sys.argv:return s,kind,json.loads((ROOT/f'{s}_{kind}.json').read_text()),None
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),context=ssl.create_default_context(cafile=certifi.where()),timeout=12) as f:d=json.load(f)['chart']['result'][0]
  obj={'source':u,'retrieved_at':datetime.datetime.now(NY).isoformat(),'data':d};(ROOT/f'{s}_{kind}.json').write_text(json.dumps(obj));return s,kind,obj,None
 except Exception as e:return s,kind,None,str(e)
data={};errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
 for s,k,d,e in ex.map(fetch,[(s,k) for s in symbols for k in ['daily','intraday']]):
  if e:errors.append([s,k,e])
  else:data[s,k]=d
def frame(s,k):
 d=data[s,k]['data'];assert d['meta']['dataGranularity']==('1d' if k=='daily' else '1m');f=pd.DataFrame(d['indicators']['quote'][0],index=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert(NY))
 if k=='daily':f['adjusted']=d['indicators']['adjclose'][0]['adjclose'];f=f[f.index.date<now.date()];f.index=f.index.normalize()
 else:f=f[(f.index.date==now.date())&(f.index.hour*60+f.index.minute>=570)&(f.index.hour*60+f.index.minute<960)]
 return f.dropna(subset=['open','high','low','close'])
good=[s for s in symbols if (s,'daily') in data and (s,'intraday') in data]
intr={s:frame(s,'intraday') for s in good};daily={s:frame(s,'daily') for s in good}
cut=min(f.index[-1].floor('min') for f in intr.values())
rows=[]
for s in good:
 f=intr[s];f=f[f.index<cut];d=daily[s];c=d.close;p=float(f.close.iloc[-1]);prev=float(c.iloc[-1]);v=float(f.volume.sum());vw=float((((f.high+f.low+f.close)/3)*f.volume).sum()/v)
 delta=c.diff().dropna();g=delta.clip(lower=0);l=(-delta).clip(lower=0);ag=g.iloc[:14].mean();al=l.iloc[:14].mean()
 for x,y in zip(g.iloc[14:],l.iloc[14:]):ag=(ag*13+x)/14;al=(al*13+y)/14
 rsi=100-100/(1+ag/al) if al else (100 if ag else 50)
 tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1).iloc[1:];atr=tr.iloc[:14].mean()
 for x in tr.iloc[14:]:atr=(atr*13+x)/14
 ret=np.log(d.adjusted).diff();qret=np.log(daily['QQQ'].adjusted).diff();pair=pd.concat([ret.rename('s'),qret.rename('q')],axis=1).dropna().tail(60);beta=float(pair.cov().loc['s','q']/pair.q.var())
 m=data[s,'intraday']['data']['meta'];qq=daily['QQQ'].adjusted
 rows.append({'symbol':s,'price':p,'quote':m['regularMarketPrice'],'quote_time':datetime.datetime.fromtimestamp(m['regularMarketTime'],NY).isoformat(),'prev':prev,'change_pct':100*(p/prev-1),'open':float(f.open.iloc[0]),'high':float(f.high.max()),'low':float(f.low.min()),'vwap':vw,'vwap_distance_pct':100*(p/vw-1),'location':float((p-f.low.min())/(f.high.max()-f.low.min())),'open_return_pct':100*(p/f.open.iloc[0]-1),'volume':v,'avg_daily_volume20':float(d.volume.tail(20).mean()),'ma20':float(c.tail(20).mean()),'ma5':float(c.tail(5).mean()),'ma50':float(c.tail(50).mean()),'return5_pct':100*(d.adjusted.iloc[-1]/d.adjusted.iloc[-6]-1),'return20_pct':100*(d.adjusted.iloc[-1]/d.adjusted.iloc[-21]-1),'relative20_vs_qqq_pp':100*((d.adjusted.iloc[-1]/d.adjusted.iloc[-21])-(qq.iloc[-1]/qq.iloc[-21])),'rsi14':rsi,'atr14':float(atr),'atr_pct':float(100*atr/prev),'daily_rv20_pct':float(ret.tail(20).std()*100),'beta_qqq60':beta,'beta_n':len(pair),'prev_high':float(d.high.iloc[-1]),'prev_low':float(d.low.iloc[-1]),'prior20_high':float(d.high.tail(20).max())})
q=next(r for r in rows if r['symbol']=='QQQ')
for r in rows:
 r['relative_today_pp']=r['change_pct']-q['change_pct'];r['beta_adjusted_today_pp']=r['change_pct']-r['beta_qqq60']*q['change_pct']
 r['gates']={'above_vwap':r['price']>r['vwap'],'positive_from_open':r['open_return_pct']>0,'outperform_qqq':r['relative_today_pp']>0,'above_ma20':r['price']>r['ma20'],'positive_relative20':r['relative20_vs_qqq_pp']>0}
 r['gates']={k:bool(v) for k,v in r['gates'].items()};r['gates_passed']=sum(r['gates'].values())
 r['chase_flag']=bool(r['vwap_distance_pct']>2 or r['rsi14']>70)
 r['eligible']=r['symbol'] in stocks and r['gates_passed']==5 and not r['chase_flag']
rows.sort(key=lambda r:(r['eligible'],r['gates_passed'],r['relative_today_pp']),reverse=True)
out={'as_of':now.isoformat(),'common_completed_cutoff_exclusive':str(cut),'technical_as_of':'2026-09-09','rows':rows,'errors':errors,'note':'Gate count is a rule checklist, not opportunity score or confidence probability. VWAP is minute typical-price approximation. Same-clock cumulative volume ratios and order flows not available in initial scan.'}
(ROOT/'筛选结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for r in rows:print(' '.join([r['symbol'],f"p={r['price']:.2f}",f"chg={r['change_pct']:.2f}%",f"VWAP={r['vwap']:.2f}",f"loc={r['location']:.2f}",f"RS={r['relative_today_pp']:.2f}",f"R20={r['return20_pct']:.1f}",f"RSI={r['rsi14']:.1f}",f"ATR%={r['atr_pct']:.1f}",f"gates={r['gates_passed']}",f"chase={r['chase_flag']}",f"eligible={r['eligible']}"]))
print('ERRORS',errors,'CUTOFF',cut)
inp=json.loads((ROOT.parents[1]/'量化参考/数据输入模板.json').read_text());inp['as_of']=datetime.datetime.now(NY).isoformat();(ROOT/'量化输入.json').write_text(json.dumps(inp,ensure_ascii=False,indent=2));sys.path.insert(0,str(ROOT.parents[1]/'量化参考'))
from 计算评分 import calculate
(ROOT/'量化覆盖率.json').write_text(json.dumps(calculate(inp),ensure_ascii=False,indent=2))
