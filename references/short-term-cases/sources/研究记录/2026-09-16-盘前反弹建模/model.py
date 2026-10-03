"""Auditable descriptive rebound study; no orders or calibrated forecasts."""
import json, ssl, urllib.request, math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import certifi

P=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
TODAY='2026-09-16'
SYMBOLS='SPY QQQ IWM RSP SMH SOXX XLK XLF XLE XLV XLI XLU XLP XLY XLC XLRE XLB TLT HYG NVDA AMD AVGO MU SNDK ORCL CRWV MSFT GOOGL AMZN META TSLA TSLL VRT ^VIX ^TNX BZ=F DX-Y.NYB'.split()

def fetch(s, intraday):
    tag='5m' if intraday else '1d'
    url=f'https://query2.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?range={"5d" if intraday else "5y"}&interval={tag}&includePrePost={str(intraday).lower()}&events=div%2Csplits'
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=25,context=ssl.create_default_context(cafile=certifi.where())) as r: data=json.load(r)
    payload={'retrieved_at':datetime.now(NY).isoformat(),'url':url,'response':data}
    (P/'raw'/f'{s}-{tag}.json').write_text(json.dumps(payload))
    return s,tag,data

def frame(data):
    j=data['chart']['result'][0]
    f=pd.DataFrame(j['indicators']['quote'][0],index=pd.to_datetime(j['timestamp'],unit='s',utc=True).tz_convert(NY)).dropna(subset=['close'])
    if 'adjclose' in j['indicators']:
        f['adj']=pd.Series(j['indicators']['adjclose'][0]['adjclose'],index=pd.to_datetime(j['timestamp'],unit='s',utc=True).tz_convert(NY)).reindex(f.index)
    f['date']=f.index.strftime('%Y-%m-%d')
    return f,j['meta']

def wilson(w,n):
    if not n:return None
    p=w/n;z=1.96;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [100*(c-h),100*(c+h)]

def stats(x):
    x=x.dropna();n=len(x)
    return {'n':n,'positive_pct':100*float((x>0).mean()) if n else None,'wilson95_pct':wilson(int((x>0).sum()),n),'mean_pct':100*float(x.mean()) if n else None,'median_pct':100*float(x.median()) if n else None,'p10_pct':100*float(x.quantile(.1)) if n else None}

def analyze(data):
    daily={};out={}
    for s in SYMBOLS:
        if (s,'1d') not in data:continue
        f,meta=frame(data[s,'1d']);f=f[f.date<TODAY].copy();f.index=f.date
        if len(f)<25:continue
        daily[s]=f
        c=f.adj;ret=c.pct_change();last=f.iloc[-1];tr=pd.concat([f.high-f.low,(f.high-f.close.shift()).abs(),(f.low-f.close.shift()).abs()],axis=1).max(axis=1)
        delta=c.diff();gain=delta.clip(lower=0).iloc[1:15].mean();loss=(-delta.clip(upper=0)).iloc[1:15].mean()
        for v in delta.iloc[15:]:gain=(gain*13+max(v,0))/14;loss=(loss*13+max(-v,0))/14
        rsi=50 if gain==loss==0 else 100 if loss==0 else 100-100/(1+gain/loss)
        d={'date':last.date,'close':float(last.close),'high':float(last.high),'low':float(last.low),'r1_pct':100*ret.iloc[-1],'r5_pct':100*(c.iloc[-1]/c.iloc[-6]-1),'r20_pct':100*(c.iloc[-1]/c.iloc[-21]-1),'ma20_price_basis':c.tail(20).mean()*last.close/c.iloc[-1],'ma50_price_basis':c.tail(50).mean()*last.close/c.iloc[-1],'rsi14':rsi,'atr14_sma':tr.tail(14).mean(),'volume_ratio20':last.volume/f.volume.iloc[-21:-1].mean() if f.volume.iloc[-21:-1].mean()>0 else None}
        if (s,'5m') in data:
            bars,m=frame(data[s,'5m']);pre=bars[(bars.date==TODAY)&(bars.index.hour<9)|((bars.date==TODAY)&(bars.index.hour==9)&(bars.index.minute<30))]
            if len(pre):
                d['premarket']={'bar_time':pre.index[-1].isoformat(),'price':pre.close.iloc[-1],'change_pct':100*(pre.close.iloc[-1]/last.close-1),'high':pre.high.max(),'low':pre.low.min(),'volume':int(pre.volume.sum()),'volume_usable':bool(pre.volume.sum()>0),'extremes_validated':False}
        out[s]=d
    qret=daily['QQQ'].adj.pct_change()
    for s,f in daily.items():
        a=pd.concat([f.adj.pct_change().rename('s'),qret.rename('q')],axis=1).dropna().tail(60)
        beta=a.s.cov(a.q)/a.q.var();alpha=a.s.mean()-beta*a.q.mean();res=a.s-alpha-beta*a.q
        out[s].update(beta60_qqq=beta,residual_daily_std_pct=100*res.std(ddof=2),rs20_qqq_pp=out[s]['r20_pct']-out['QQQ']['r20_pct'])
    analog={}
    for s in ['SPY','QQQ','SMH']:
        f=daily[s];r=f.adj.pct_change();gap=f.open/f.close.shift()-1;oc=f.close/f.open-1
        setup=(r.shift(1)<0)&(r.shift(2)<0)&(f.adj.shift(1)/f.adj.shift(6)-1<0)&(gap>0)&(gap<=.015)
        selected=f.index[setup]
        recent=selected[selected>='2025-09-16'];early=selected[selected<'2025-09-16']
        analog[s]={'current_prior_close_setup_matches':bool(r.iloc[-1]<0 and r.iloc[-2]<0 and f.adj.iloc[-1]/f.adj.iloc[-6]-1<0),'today_opening_gap_confirmed':False,'definition':'Prior two close-to-close returns <0, prior 5-day return <0, actual next opening gap (0,1.5%]. Excludes today. Descriptive, not FOMC matched. Unadjusted intraday OHLC ratios; corporate-action boundary caveat.','all':stats(oc.loc[selected]),'earlier':stats(oc.loc[early]),'recent_year':stats(oc.loc[recent]),'unconditional_oc':stats(oc),'dates':list(selected),'previous_close_revisited_pct':100*float((f.low.loc[selected]<=f.close.shift().loc[selected]).mean())}
    ts=daily['TSLA'];gaps=[]
    for i in range(1,len(ts)):
        prev=ts.iloc[i-1];now=ts.iloc[i];sub=ts.iloc[i:]
        if now.open>prev.close:
            lo=prev.close;hi=min(now.open,sub.low.min())
            if hi>lo:gaps.append({'date':now.date,'direction':'up','unfilled_low':lo,'unfilled_high':hi})
        elif now.open<prev.close:
            lo=max(now.open,sub.high.max());hi=prev.close
            if hi>lo:gaps.append({'date':now.date,'direction':'down','unfilled_low':lo,'unfilled_high':hi})
    result={'created_at':datetime.now(NY).isoformat(),'universe':SYMBOLS,'snapshot':out,'analog':analog,'tsla_unfilled_open_close_gaps_5y':gaps,'limitations':['No calibrated forward probability; analog rules selected for current setup and not FOMC-specific.','Current universe not used for historical portfolio backtest.','Premarket prices may be delayed; 5m timestamp is bar start, not last trade.','ATR is simple 14-day average, RSI is Wilder; beta estimates use last 60 daily total returns.']}
    (P/'model-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':
    import sys,urllib.parse
    (P/'raw').mkdir(exist_ok=True)
    data={}
    if '--offline' in sys.argv:
        for s in SYMBOLS:
            for tag in ['1d','5m']:
                p=P/'raw'/f'{s}-{tag}.json'
                if p.exists():data[s,tag]=json.loads(p.read_text())['response']
    else:
        with ThreadPoolExecutor(max_workers=6) as pool:
            jobs=[pool.submit(fetch,s,intraday) for s in SYMBOLS for intraday in [False,True]]
            for task in as_completed(jobs):
                try:s,t,j=task.result();data[s,t]=j
                except Exception as e:print('FETCH ERROR',repr(e),file=sys.stderr)
    analyze(data)
