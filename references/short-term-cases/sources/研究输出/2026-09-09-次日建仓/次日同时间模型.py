"""Same-clock event comparison adapted from local next-session event studies; no order execution."""
import json,datetime,pathlib,ssl,certifi,urllib.request,concurrent.futures,math
import numpy as np
from zoneinfo import ZoneInfo
R=pathlib.Path(__file__).parent;C=pathlib.Path('/private/tmp/nextday-20260909');C.mkdir(exist_ok=True);D=pathlib.Path('/private/tmp/stockscan-20260909');NY=ZoneInfo('America/New_York');SYMS=['CLS','WDC','ALAB','MU','CRWV','VRT'];TODAY='2026-09-09';CUT=685 # 11:25 EDT
(R/'模型规格.json').write_text(json.dumps({'created_at':datetime.datetime.now(NY).isoformat(),'symbols':SYMS,'cutoff':'11:25 EDT','features':'match signs of price relative to previous close and completed-bar VWAP proxy at same clock; no threshold fitting','targets':['next session close/entry at 11:25 minus1','next session close/current session official daily close minus1'],'jump_definition':'next regular daily total return >=8%; >=10% sensitivity','eligibility':'complete current and next historical ordinary sessions, next outcome before current date','cost_scenario':'10bp entry plus10bp exit on holding return only; no stops, fills or news controls; statistical frequencies not calibrated probability'},ensure_ascii=False,indent=2))
def fetch(s):
 u=f'https://query2.finance.yahoo.com/v8/finance/chart/{s}?range=60d&interval=5m&includePrePost=false'
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),context=ssl.create_default_context(cafile=certifi.where()),timeout=25) as f:x=json.load(f)['chart']['result'][0]
  (C/f'{s}.json').write_text(json.dumps(x));return s,x
 except Exception as e:return s,{'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p:raw=dict(p.map(fetch,SYMS))
def wilson(k,n):
 if not n:return None
 p=k/n;z=1.96;den=1+z*z/n;m=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [m-h,m+h]
def stats(a):
 if not a:return {'n':0}
 ret=np.array([v['holding_net'] for v in a]);daily=np.array([v['next_daily_return'] for v in a]);n=len(a);k=int(sum(daily>=.08))
 return {'n':n,'holding_up_count':int(sum(ret>0)),'holding_mean_pct':float(ret.mean()*100),'holding_median_pct':float(np.median(ret)*100),'holding_p10_p90_pct':(np.quantile(ret,[.1,.9])*100).tolist(),'holding_gain8_count':int(sum(ret>=.08)),'holding_loss5_count':int(sum(ret<=-.05)),'next_daily_gain8_count':k,'next_daily_gain10_count':int(sum(daily>=.10)),'next_daily_gain8_wilson95':wilson(k,n),'dates':[v['date'] for v in a]}
out={}
for s,x in raw.items():
 if 'error' in x:out[s]=x;continue
 q=x['indicators']['quote'][0];days={}
 for i,t in enumerate(x['timestamp']):
  dt=datetime.datetime.fromtimestamp(t,NY);minute=dt.hour*60+dt.minute
  if dt.second or minute%5 or not 570<=minute<960 or any(q[k][i] is None for k in ['open','high','low','close']):continue
  days.setdefault(str(dt.date()),[]).append({'minute':minute,**{k:q[k][i] for k in ['open','high','low','close','volume']}})
 dr=json.loads((D/f'{s}.json').read_text());dq=dr['indicators']['quote'][0];da=dr['indicators']['adjclose'][0]['adjclose'];daily={}
 for i,t in enumerate(dr['timestamp']):
  d=str(datetime.datetime.fromtimestamp(t,NY).date())
  if dq['close'][i] is not None and da[i] is not None:daily[d]={'c':dq['close'][i],'a':da[i]}
 dates=sorted(daily);previous={dates[i]:dates[i-1] for i in range(1,len(dates))};nextdate={dates[i]:dates[i+1] for i in range(len(dates)-1)}
 def point(d):
  b=[z for z in days[d] if z['minute']<CUT];assert len(b)==(CUT-570)//5
  vol=sum(z['volume'] or 0 for z in b);vwap=sum((z['high']+z['low']+z['close'])/3*(z['volume'] or 0) for z in b)/vol;p=b[-1]['close'];return p,vwap,p>daily[previous[d]]['c'],p>vwap
 try:cur=point(TODAY)
 except Exception as e:out[s]={'error':'incomplete current snapshot '+str(e)};continue
 rows=[]
 for d in sorted(days):
  if d>=TODAY or d not in nextdate or d not in previous:continue
  nd=nextdate[d]
  if nd>=TODAY or nd not in days:continue
  if len(days[d])!=78 or len(days[nd])!=78:continue
  try:p,v,up,above=point(d)
  except:continue
  entry=p*daily[d]['a']/daily[d]['c'];holding=daily[nd]['a']*(1-.001)/(entry*(1+.001))-1;ndret=daily[nd]['a']/daily[d]['a']-1
  rows.append({'date':d,'nextdate':nd,'up_vs_prev':up,'above_vwap':above,'holding_net':holding,'next_daily_return':ndret})
 matched=[v for v in rows if (v['up_vs_prev'],v['above_vwap'])==cur[2:]]
 out[s]={'quote':x['meta']['regularMarketPrice'],'quote_time':datetime.datetime.fromtimestamp(x['meta']['regularMarketTime'],NY).isoformat(),'current_1125':dict(price=cur[0],vwap_proxy=cur[1],up_vs_previous=cur[2],above_vwap=cur[3]),'all':stats(rows),'matched':stats(matched),'rows':rows}
(R/'次日模型结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for s,v in out.items():
 if 'error' in v:print(s,v);continue
 for k in ['all','matched']:v[k].pop('dates',None)
 v.pop('rows');print(s,json.dumps(v,ensure_ascii=False))
