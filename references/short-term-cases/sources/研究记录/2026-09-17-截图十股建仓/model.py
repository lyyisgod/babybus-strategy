"""Exploratory factors, point-in-time price signals; no calibrated probabilities.
Run online to save Yahoo responses; --offline replays the saved snapshot.
"""
import json, ssl, sys, urllib.request
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import certifi
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
NOW=datetime.now(timezone.utc)
OFFLINE='--offline' in sys.argv
if OFFLINE: NOW=datetime.fromisoformat(json.loads((ROOT/'manifest.json').read_text())['asof'])
POOL='IOVA ABCL TXG CDNA TEAM ATRC AMPL UTZ PBF RNG'.split()
SYMBOLS=POOL+'SNDK SPY QQQ XBI IGV XHE XLP CRAK'.split()
BENCH=dict(IOVA='XBI',ABCL='XBI',TXG='XBI',CDNA='XBI',TEAM='IGV',ATRC='XHE',AMPL='IGV',UTZ='XLP',PBF='CRAK',RNG='IGV')
CTX=ssl.create_default_context(cafile=certifi.where())
def fetch(job):
    s,kind=job
    query='range=5y&interval=1d&events=div%2Csplits' if kind=='daily' else 'range=5d&interval=5m&includePrePost=true'
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?{query}'
    path=ROOT/'raw'/f'{s}-{kind}.json'
    if OFFLINE:return {'symbol':s,'kind':kind,'source':url,'replay':True}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,context=CTX,timeout=30) as r:d=json.load(r)
        if not d['chart'].get('result'):raise ValueError(str(d))
        path.write_text(json.dumps(d))
        return {'symbol':s,'kind':kind,'source':url,'fetched_at':datetime.now(timezone.utc).isoformat()}
    except Exception as e:return {'symbol':s,'kind':kind,'source':url,'error':str(e)}
def history(s,kind='daily'):
    x=json.loads((ROOT/'raw'/f'{s}-{kind}.json').read_text())['chart']['result'][0]
    ix=pd.to_datetime(x['timestamp'],unit='s',utc=True).tz_convert(NY)
    d=pd.DataFrame(x['indicators']['quote'][0],index=ix).dropna(subset=['close'])
    if kind=='daily':
        d['adj']=pd.Series(x['indicators']['adjclose'][0]['adjclose'],index=ix)
        d.index=pd.Index(d.index.date)
        d=d.loc[d.index<NOW.astimezone(NY).date()].dropna(subset=['adj'])
    return d,x['meta']
def wilder(s,n=14):
    a=np.full(len(s),np.nan)
    if len(s)>=n:
        a[n-1]=np.mean(s.iloc[:n])
        for i in range(n,len(s)):a[i]=(a[i-1]*(n-1)+s.iloc[i])/n
    return pd.Series(a,index=s.index)
def factors(d):
    d=d.copy();c=d.close; a=d.adj; ratio=a/c
    # Consistent adjustment of OHLC for indicators and returns.
    h=d.high*ratio; l=d.low*ratio
    tr=pd.concat([h-l,(h-a.shift()).abs(),(l-a.shift()).abs()],axis=1).max(axis=1)
    f=pd.DataFrame(index=d.index)
    f['price']=c
    for n in [10,20,50]:f[f'ema{n}']=a.ewm(span=n,adjust=False).mean()/ratio
    gain=wilder(a.diff().dropna().clip(lower=0)); loss=wilder(-a.diff().dropna().clip(upper=0))
    f['rsi14']=100-100/(1+gain/loss)
    f.loc[(loss==0).reindex(f.index,fill_value=False),'rsi14']=100
    f.loc[((gain==0)&(loss==0)).reindex(f.index,fill_value=False),'rsi14']=50
    f['atr14']=wilder(tr)/ratio
    f['atr_pct']=100*f.atr14/c
    for n in [20,63,126]:f[f'ret{n}']=a.pct_change(n)*100
    f['volume_ratio']=d.volume/d.volume.shift().rolling(20).mean()
    f['dollar_volume20_m']=(c*d.volume).rolling(20).mean()/1e6
    f['prior20high']=h.shift().rolling(20).max()/ratio
    f['prior252high']=h.shift().rolling(252,min_periods=200).max()/ratio
    f['high252_inclusive']=h.rolling(252,min_periods=200).max()/ratio
    f['dist52high_pct']=100*(c/f.prior252high-1)
    f['extension20_atr']=(c-f.ema20)/f.atr14
    f['above_stack']=(c>f.ema10)&(f.ema10>f.ema20)&(f.ema20>f.ema50)
    f['breakout']=c>f.prior20high
    f['close_highs20']=(a>a.shift().rolling(252,min_periods=200).max()).rolling(20).sum()
    f['prior10low']=l.shift().rolling(10).min()/ratio
    return f
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def main():
    (ROOT/'raw').mkdir(exist_ok=True)
    if not OFFLINE:(ROOT/'universe.json').write_text(json.dumps({'asof':NOW.isoformat(),'pool':POOL,'benchmarks':BENCH,'note':'Current screenshot-selected surviving stocks; selection bias. Sector ETFs are imperfect proxies.'},indent=2))
    jobs=[(s,'daily') for s in SYMBOLS]+[(s,'intraday') for s in POOL+['SNDK','SPY','QQQ']]
    with ThreadPoolExecutor(max_workers=6) as ex:manifest=list(ex.map(fetch,jobs))
    if not OFFLINE:(ROOT/'manifest.json').write_text(json.dumps({'asof':NOW.isoformat(),'requests':manifest},indent=2))
    errors={};data={};fs={}
    for s in SYMBOLS:
        try:data[s]=history(s)[0];fs[s]=factors(data[s])
        except Exception as e:errors[s]=str(e)
    rows=[]; quotes=[]; analogs=[]
    for s in POOL+['SNDK','SPY','QQQ']:
        if s not in data:continue
        d=data[s]; f=fs[s];row={'symbol':s,'daily_date':str(d.index[-1]),**f.iloc[-1].to_dict()}
        for b in ['SPY',BENCH.get(s,'QQQ')]:
            if b in data:
                joint=pd.concat([d.adj.rename('s'),data[b].adj.rename('b')],axis=1).dropna()
                row[f'rs20_{b}']=100*((joint.s.iloc[-1]/joint.s.iloc[-21]-1)-(joint.b.iloc[-1]/joint.b.iloc[-21]-1))
        rows.append(row)
        try:
            intr,m=history(s,'intraday'); stamp=datetime.fromtimestamp(m['regularMarketTime'],NY)
            today=intr.loc[(intr.index.date==NOW.astimezone(NY).date())&(intr.index.hour*60+intr.index.minute>=570)&(intr.index.hour*60+intr.index.minute<960)]
            p=float(m['regularMarketPrice']);prev=float(d.close.iloc[-1]); complete=today.loc[today.index+pd.Timedelta(minutes=5)<=pd.Timestamp(NOW).tz_convert(NY)]
            typical=(complete.high+complete.low+complete.close)/3
            vwap=float((typical*complete.volume).sum()/complete.volume.sum()) if complete.volume.sum()>0 else None
            historical=[]
            if len(complete):
                endmin=complete.index[-1].hour*60+complete.index[-1].minute
                reg=intr.loc[(intr.index.hour*60+intr.index.minute>=570)&(intr.index.hour*60+intr.index.minute<=endmin)]
                for day,g in reg.groupby(reg.index.date):
                    if day<NOW.astimezone(NY).date():historical.append(float(g.volume.sum()))
            quotes.append(dict(symbol=s,time_et=stamp.isoformat(),price=p,prev_close=prev,change_pct=100*(p/prev-1),open=m.get('regularMarketOpen',float(today.open.iloc[0]) if len(today) else None),high=m.get('regularMarketDayHigh'),low=m.get('regularMarketDayLow'),volume=m.get('regularMarketVolume'),vwap_5m_proxy=vwap,same_time_volume_ratio=float(complete.volume.sum()/np.mean(historical)) if historical and np.mean(historical)>0 else None,comparison_days=len(historical),last_complete_bar=str(complete.index[-1]) if len(complete) else None))
        except Exception as e:errors[s+'-intraday']=str(e)
        if s not in POOL:continue
        # Predeclared descriptive analogue: screenshot trend + near highs; next open execution.
        # No parameter fitting. Only fully observed 63-session paths. Per-stock events don't overlap.
        last=-1000
        for i in range(252,len(d)-64):
            z=f.iloc[i]
            if i-last<64 or not z.above_stack or not -10<=z.dist52high_pct<=5:continue
            entry=d.open.iloc[i+1]*(d.adj.iloc[i+1]/d.close.iloc[i+1])
            path=d.adj.iloc[i+1:i+64]; gain=100*(path.iloc[-1]/entry-1)
            nh=int((path>d.adj.rolling(252,min_periods=200).max().shift().iloc[i+1:i+64]).sum())
            analogs.append(dict(symbol=s,signal_date=str(d.index[i]),entry_date=str(d.index[i+1]),end_date=str(d.index[i+63]),ret63_net_pct=gain-0.2,mae_close_pct=100*(path.min()/entry-1),new_52w_closes=nh,trend_success=bool(gain-0.2>=30 and nh>=5),return_positive=bool(gain-0.2>0)))
            last=i
    df=pd.DataFrame(rows);df.to_csv(ROOT/'factors.csv',index=False)
    pd.DataFrame(quotes).to_csv(ROOT/'quotes.csv',index=False)
    pd.DataFrame(analogs).to_csv(ROOT/'historical_analogues.csv',index=False)
    # Exploratory execution gate, designed after reviewing current data; NOT backtested.
    # Limits are research choices, never a success probability or an automatic order.
    gates=[]
    for q in quotes:
        s=q['symbol']
        if s not in POOL:continue
        z=fs[s].iloc[-1]; latest_extension=(q['price']-z.ema20)/z.atr14
        rr=next(r for r in rows if r['symbol']==s)
        conditions=dict(above_yesterday_52w_high=q['price']>z.high252_inclusive,
                        above_vwap=q['vwap_5m_proxy'] is not None and q['price']>q['vwap_5m_proxy'],
                        volume_ge_1_2=q['same_time_volume_ratio'] is not None and q['same_time_volume_ratio']>=1.2,
                        extension_le_3atr=latest_extension<=3,
                        positive_sector_rs=rr.get(f'rs20_{BENCH[s]}',float('nan'))>0,
                        liquid=z.dollar_volume20_m>=10,
                        no_takeprivate=s!='UTZ')
        gates.append(dict(symbol=s,extension_now_atr=latest_extension,**conditions,all_pass=all(conditions.values())))
    pd.DataFrame(gates).to_csv(ROOT/'entry_gates.csv',index=False)
    summary={'asof':NOW.isoformat(),'errors':errors,'analogue_n':len(analogs),'analogue_trend_success':sum(x['trend_success'] for x in analogs),'warning':'Descriptive, survivorship and screenshot selection bias; correlated symbols. Not calibrated current-stock probabilities. Historical signal thresholds fixed before first run; high definition corrected from ATH to rolling 52w during QA. Entry gates designed after inspecting current data and not validated. No fit; cost 20 bp round trip; no stop modeled.'}
    (ROOT/'summary.json').write_text(json.dumps(clean(summary),indent=2))
    print(df[['symbol','price','ret20','ret63','rsi14','volume_ratio','extension20_atr','dist52high_pct','close_highs20','atr14','prior20high','prior10low']].round(3).to_string(index=False))
    print(pd.DataFrame(quotes).round(3).to_string(index=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
