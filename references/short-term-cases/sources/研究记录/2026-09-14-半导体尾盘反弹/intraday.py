"""Intraday diagnostic, not a predictive or causal model. Public data only."""
import json, ssl, urllib.request, urllib.parse, datetime, statistics, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parent
TZ=ZoneInfo('America/New_York')
NOW=datetime.datetime.now(datetime.timezone.utc)
if '--offline' in sys.argv:
 NOW=datetime.datetime.fromisoformat(json.loads((R/'results.json').read_text())['observed_at'])
HARDWARE=['NVDA','AMD','AVGO','MU','TSM','MRVL','ASML','AMAT','LRCX','KLAC','ON','TXN','ADI','NXPI','INTC','ARM','SNDK','ALAB','CRDO','QCOM','STM','TER']
SYMBOLS=HARDWARE+['SMH','SOXX','QQQ','SPY','IGV','CIBR','MSFT','^TNX','BZ=F','^VIX']
def location(p,lo,hi):return (p-lo)/(hi-lo) if hi>lo else None
def get(s):
 u='https://query1.finance.yahoo.com/v8/finance/chart/'+urllib.parse.quote(s,safe='')+'?range=5d&interval=5m'
 try:
  raw_path=R/(s.replace('^','INDEX_').replace('=','_')+'-raw.json')
  raw=json.loads(raw_path.read_text()) if '--offline' in sys.argv else json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),context=ssl._create_unverified_context(),timeout=15))
  if '--offline' not in sys.argv:raw_path.write_text(json.dumps(raw))
  x=raw['chart']['result'][0];m=x['meta'];q=x['indicators']['quote'][0]
  rows=[]
  for i,t in enumerate(x['timestamp']):
   if q['close'][i] is None:continue
   dt=datetime.datetime.fromtimestamp(t,datetime.timezone.utc).astimezone(TZ)
   if not 570<=dt.hour*60+dt.minute<960:continue
   if dt.astimezone(datetime.timezone.utc)>NOW:continue
   rows.append(dict(dt=dt,**{k:v[i] for k,v in q.items()}))
  today=[z for z in rows if z['dt'].date()==NOW.astimezone(TZ).date()]
  # Use only completed five-minute bars for cross-sectional comparisons.
  done=[z for z in today if z['dt'].minute%5==0 and z['dt']+datetime.timedelta(minutes=5)<=NOW]
  if not done:raise ValueError('no complete current bars')
  p=done[-1]['close'];lo=min(z['low'] for z in done);hi=max(z['high'] for z in done)
  vs=sum(z['volume'] or 0 for z in done)
  vw=sum((z['high']+z['low']+z['close'])/3*(z['volume'] or 0) for z in done)/vs if vs else None
  prev=m.get('previousClose')
  if prev is None and m.get('regularMarketChangePercent') is not None:prev=m['regularMarketPrice']/(1+m['regularMarketChangePercent']/100)
  if prev is None:prev=[z for z in rows if z['dt'].date()<done[-1]['dt'].date()][-1]['close']
  t0=done[-1]['dt']-datetime.timedelta(minutes=60)
  base=[z for z in done if z['dt']<=t0][-1]
  recent=[z for z in done if z['dt']>t0]
  minutes={z['dt'].hour*60+z['dt'].minute for z in recent}
  past={}
  for z in rows:
   if z['dt'].date()<done[-1]['dt'].date() and z['dt'].hour*60+z['dt'].minute in minutes:past.setdefault(str(z['dt'].date()),[]).append(z)
  volumes=[sum(z['volume'] or 0 for z in a) for a in past.values() if len(a)==len(recent)]
  return {'symbol':s,'bar_start_et':str(done[-1]['dt']),'latest':m['regularMarketPrice'],'quote_time_et':str(datetime.datetime.fromtimestamp(m['regularMarketTime'],TZ)),'price':p,'previous_close':prev,'day_pct':100*(p/prev-1),'low':lo,'high':hi,'low_time_et':str(min(done,key=lambda z:z['low'])['dt']),'bounce_pct':100*(p/lo-1),'recovered_selloff_fraction':(p-lo)/(prev-lo) if prev>lo else None,'range_location':location(p,lo,hi),'vwap_proxy':vw,'above_vwap':p>vw if vw else None,'last60_pct':100*(p/base['close']-1),'last60_volume_ratio_vs_previous_days':sum(z['volume'] or 0 for z in recent)/statistics.mean(volumes) if volumes and statistics.mean(volumes)>0 else None,'volume_comparison_days':len(volumes),'source':u}
 except Exception as e:return {'symbol':s,'error':str(e)}
assert location(105,100,110)==.5 and location(100,100,100) is None
with ThreadPoolExecutor(max_workers=6) as ex:data=list(ex.map(get,SYMBOLS))
valid={z['symbol']:z for z in data if 'error' not in z}
if 'QQQ' in valid:
 for z in valid.values():z['last60_vs_qqq_pp']=z['last60_pct']-valid['QQQ']['last60_pct'] if z['symbol'] not in ['^TNX','BZ=F','^VIX'] and z['bar_start_et']==valid['QQQ']['bar_start_et'] else None
hw=[valid[s] for s in HARDWARE if s in valid]
summary={'hardware_valid':len(hw),'hardware_total':len(HARDWARE),'above_vwap_count':sum(z['above_vwap'] is True for z in hw),'positive_last60_count':sum(z['last60_pct']>0 for z in hw),'median_bounce_pct':statistics.median(z['bounce_pct'] for z in hw),'median_last60_vs_qqq_pp':statistics.median(z['last60_vs_qqq_pp'] for z in hw)} if hw else {}
out={'observed_at':NOW.isoformat(),'hardware_universe_fixed_before_fetch':HARDWARE,'method':'Completed 5m regular-session bars; VWAP uses typical-price proxy; equal-weight hardware diagnostic, no prediction calibration or causal identification; four preceding sessions maximum for volume comparison. Yields and futures are context, not portfolio constituents.','summary':summary,'rows':data}
(R/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False))
