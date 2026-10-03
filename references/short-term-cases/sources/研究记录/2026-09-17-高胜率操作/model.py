"""Fixed, auditable LOC screen; discovery/validation shortlist before test release."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist
import hashlib,json,math,ssl,sys,urllib.request
import numpy as np
import pandas as pd
import certifi

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
NY=ZoneInfo('America/New_York'); TODAY=pd.Timestamp(P['date'],tz=NY)
COST=P['cost_round_trip']; RULES=list(P['strategies']); H=20

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def save(name,v):
    (ROOT/name).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False))
def fetch_one(s):
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=10y&interval=1d&events=div%2Csplits'
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        raw=urllib.request.urlopen(req,context=ssl.create_default_context(cafile=certifi.where()),timeout=30).read()
        r=json.loads(raw)['chart']['result'][0];assert r['timestamp']
        (ROOT/'raw'/f'{s}.json').write_bytes(raw)
        return dict(symbol=s,url=url,fetched_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest())
    except Exception as e:return dict(symbol=s,url=url,error=str(e))
def fetch():
    (ROOT/'raw').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as ex: requests=list(ex.map(fetch_one,P['universe']))
    save('manifest.json',dict(asof=datetime.now(timezone.utc).isoformat(),requests=requests))
    print('FETCH',sum('error' not in r for r in requests),'/',len(requests),flush=True)
def wilder(s,n):
    v=np.full(len(s),np.nan)
    a=s.to_numpy(dtype=float)
    if len(a)>=n:
        v[n-1]=np.mean(a[:n])
        for i in range(n,len(a)):v[i]=(v[i-1]*(n-1)+a[i])/n
    return pd.Series(v,index=s.index)
def load(s):
    r=json.loads((ROOT/'raw'/f'{s}.json').read_text())['chart']['result'][0]
    ix=pd.to_datetime(r['timestamp'],unit='s',utc=True).tz_convert(NY).normalize()
    d=pd.DataFrame(r['indicators']['quote'][0],index=ix)
    d['adj']=r['indicators']['adjclose'][0]['adjclose']
    d=d.loc[d.index<TODAY].dropna(subset=['open','high','low','close','adj','volume'])
    assert d.index.is_unique and d.index.is_monotonic_increasing
    assert ((d.high+1e-5>=d[['open','close']].max(axis=1))&(d.low-1e-5<=d[['open','close']].min(axis=1))).all(),s
    ratio=d.adj/d.close
    for col in ['open','high','low','close']:d['raw_'+col]=d[col]
    for col in ['open','high','low','close']:d[col]*=ratio
    return d,r['meta'],ratio
def factors(d):
    c=d.close
    f=pd.DataFrame(index=d.index)
    f['sma200']=c.rolling(200).mean()
    for n in [20,50,200]:f[f'ema{n}']=c.ewm(span=n,adjust=False).mean()
    gains=wilder(c.diff().dropna().clip(lower=0),2);losses=wilder(-c.diff().dropna().clip(upper=0),2)
    f['rsi2']=100-100/(1+gains/losses)
    f.loc[(losses==0).reindex(f.index,fill_value=False),'rsi2']=100
    f.loc[((gains==0)&(losses==0)).reindex(f.index,fill_value=False),'rsi2']=50
    tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
    f['atr']=wilder(tr,14)
    f['r3']=c.pct_change(3);f['r63']=c.pct_change(63)
    f['pullback']=(c>f.sma200)&(f.rsi2<=20)&(f.r3<0)
    f['reclaim']=(c>f.ema20)&(c.shift()<=f.ema20.shift())&(f.ema50>f.ema200)&(f.r63>0)
    f['breakout']=(c>d.high.shift().rolling(20).max())&(d.volume>=1.2*d.volume.shift().rolling(20).mean())&(c>f.sma200)
    f['baseline']=c>f.sma200
    return f
def exit_trade(bars,entry,atr):
    stop,target=entry-2*atr,entry+2*atr
    for j,(o,hi,lo,c) in enumerate(bars):
        if o<=stop:return j,o,'gap_stop'
        if o>=target:return j,o,'gap_target'
        if lo<=stop:return j,stop,'stop'
        if hi>=target:return j,target,'target'
    return len(bars)-1,bars[-1,3],'time'
def trades(d,f,rule,start,end):
    # Each segment starts flat. Never inspect rows after segment end.
    ix=np.flatnonzero((d.index>=pd.Timestamp(start,tz=NY))&(d.index<=pd.Timestamp(end,tz=NY)))
    if not len(ix):return pd.DataFrame()
    last=ix[-1]; busy=-1; out=[]
    # Yahoo quote OHLC is split-adjusted but excludes cash dividends. Use it
    # for executable fixed-dollar barriers. Indicators use total-return prices.
    raw_cols=['raw_open','raw_high','raw_low','raw_close']
    raw=d[raw_cols].copy() if all(c in d for c in raw_cols) else d[['open','high','low','close']].copy()
    raw.columns=['open','high','low','close'];bars=raw.to_numpy()
    ratio=d.close/raw.close
    for i in ix:
        if i<200 or i<=busy or i+1+H>last:continue
        if not f[rule].iloc[i]:continue
        e=i+1;entry=raw.close.iloc[e];atr=f.atr.iloc[i]/ratio.iloc[i]
        if entry>raw.close.iloc[i]+.5*atr:continue
        # Entry-day highs are never counted: fill is in its closing auction.
        j,px,reason=exit_trade(bars[e+1:e+H+1],entry,atr)
        ex=e+1+j;ret=px/entry-1-COST
        out.append(dict(signal=str(d.index[i].date()),entry_date=str(d.index[e].date()),exit_date=str(d.index[ex].date()),
            entry_index=e,exit_index=ex,entry=entry,atr=atr,exit=px,ret=ret,win=ret>0,reason=reason,hold=j+1))
        busy=ex
    return pd.DataFrame(out)
def wilson(k,n,z=1.959963984540054):
    if not n:return (None,None)
    p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den
    err=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,mid-err),min(1,mid+err)
def stats(t):
    if not len(t):return dict(n=0,win=None,mean=None,pf=None,win_loss_ratio=None,win_ci95=[None,None])
    r=t.ret.to_numpy();pos=r[r>0];neg=r[r<=0]
    eq=np.r_[1,np.cumprod(1+r)];dd=eq/np.maximum.accumulate(eq)-1
    return dict(n=len(r),wins=len(pos),win=len(pos)/len(r),mean=r.mean(),median=float(np.median(r)),
        pf=pos.sum()/abs(neg.sum()) if neg.sum()<0 else None,
        mean_win=pos.mean() if len(pos) else None,mean_loss=neg.mean() if len(neg) else None,
        win_loss_ratio=pos.mean()/abs(neg.mean()) if len(pos) and len(neg) and neg.mean()<0 else None,
        win_ci95=wilson(len(pos),len(r)),max_trade_loss=r.min(),closed_trade_equity_max_drawdown=dd.min(),
        median_hold=t.hold.median(),target_count=int(t.reason.isin(['target','gap_target']).sum()),stop_count=int(t.reason.isin(['stop','gap_stop']).sum()))
def discover():
    rows=[];current=[];errors={}
    for s in P['universe']:
        try:d,meta,ratio=load(s)
        except Exception as e:errors[s]=str(e);continue
        f=factors(d);quote=float(meta['regularMarketPrice']);cap=(d.close.iloc[-1]+.5*f.atr.iloc[-1])/ratio.iloc[-1]
        time=datetime.fromtimestamp(meta['regularMarketTime'],NY).isoformat()
        for rule in RULES:
            a=trades(d,f,rule,'2016-01-01','2020-12-31');b=trades(d,f,rule,'2021-01-01','2023-12-31')
            aa,bb,cc=stats(a),stats(b),stats(pd.concat([a,b],ignore_index=True))
            dev=dict(symbol=s,rule=rule,discovery=aa,validation=bb,development=cc)
            enough=aa['n']>=20 and bb['n']>=12
            eligible=bool(enough and aa['mean']>0 and bb['mean']>0 and cc['win_loss_ratio'] is not None and cc['win_loss_ratio']>=.8)
            active=bool(f[rule].iloc[-1]);price_ok=quote<=cap
            dev.update(eligible=eligible,signal_active=active,current_quote=quote,quote_time=time,loc_limit=cap,price_within_limit=price_ok,
                atr=float(f.atr.iloc[-1]/ratio.iloc[-1]),rsi2=float(f.rsi2.iloc[-1]),sma200=float(f.sma200.iloc[-1]/ratio.iloc[-1]))
            rows.append(dev)
            if active:current.append(dev)
    qualified=[r for r in rows if r['eligible'] and r['signal_active'] and r['price_within_limit']]
    qualified.sort(key=lambda r:(r['development']['win_ci95'][0],r['validation']['mean']),reverse=True)
    picks=[];seen=set()
    for r in qualified:
        if r['symbol'] in seen:continue
        picks.append(r);seen.add(r['symbol'])
        if len(picks)==3:break
    save('development.json',dict(rows=rows,errors=errors,current=current))
    lock=dict(created_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
        test_outcomes_not_read=True,selection_count=len(picks),shortlist=picks,all_current_active_count=len(current))
    # Lock only once; test evaluation cannot replace this shortlist.
    path=ROOT/'locked-shortlist.json'
    if path.exists():raise RuntimeError('Shortlist already locked; will not silently overwrite')
    save('locked-shortlist.json',lock)
    print(json.dumps(clean(lock),ensure_ascii=False,indent=2))
def block_mean_ci(r):
    rng=np.random.default_rng(9172026);n=len(r);out=[]
    for _ in range(10000):
        starts=rng.integers(0,n,size=math.ceil(n/3));ix=((starts[:,None]+np.arange(3))%n).ravel()[:n]
        out.append(r[ix].mean())
    return np.quantile(out,[.025,.975]).tolist()
def test():
    lock=json.loads((ROOT/'locked-shortlist.json').read_text())
    assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256']
    m=lock['selection_count'];z=NormalDist().inv_cdf(1-.05/max(1,m));out=[]
    for pick in lock['shortlist']:
        s,rule=pick['symbol'],pick['rule'];d,meta,ratio=load(s);f=factors(d)
        t=trades(d,f,rule,'2024-01-01','2026-09-16');r=stats(t)
        t.to_csv(ROOT/f'test-{s}-{rule}.csv',index=False)
        if len(t):
            r['mean_return_block_ci95']=block_mean_ci(t.ret.to_numpy())
            r['family_adjusted_win_lower']=wilson(r['wins'],r['n'],z)[0]
            r['last20']=stats(t.tail(20));r['annual']={str(y):stats(t.loc[t.entry_date.str.startswith(str(y))]) for y in [2024,2025,2026]}
            r['accepted']=bool(r['n']>=20 and r['win']>=.65 and r['family_adjusted_win_lower']>.5 and r['mean']>0 and r['pf'] is not None and r['pf']>=1.3 and r['win_loss_ratio'] is not None and r['win_loss_ratio']>=.8 and r['mean_return_block_ci95'][0]>0)
        else:r['accepted']=False
        base=trades(d,f,'baseline','2024-01-01','2026-09-16')
        base.to_csv(ROOT/f'baseline-{s}.csv',index=False)
        out.append(dict(symbol=s,rule=rule,test=r,baseline=stats(base),development=pick))
    save('test-results.json',dict(evaluated_at=datetime.now(timezone.utc).isoformat(),family_count=m,results=out))
    print(json.dumps(clean(out),ensure_ascii=False,indent=2))
def checks():
    z=exit_trade(np.array([[100,105,95,102.]]),100,2)
    assert z[2]=='stop' and z[1]==96
    assert exit_trade(np.array([[95,105,90,102.]]),100,2)[1]==95
    assert exit_trade(np.array([[105,106,94,102.]]),100,2)[2]=='gap_target'
    n=280;c=np.ones(n)*100;d=pd.DataFrame(dict(open=c,high=c+1,low=c-1,close=c,volume=c),index=pd.date_range('2019-01-01',periods=n,tz=NY))
    f=factors(d);assert np.isclose(f.rsi2.iloc[-1],50) and np.isclose(f.atr.iloc[-1],2)
    # Deliberate entry-session fake +100% high must not create a target exit.
    f['pullback']=False;f.iloc[210,f.columns.get_loc('pullback')]=True
    d.iloc[211,d.columns.get_loc('high')]=200
    t=trades(d,f,'pullback','2019-01-01','2020-01-01')
    assert len(t)==1 and t.iloc[0].reason=='time' and np.isclose(t.iloc[0].ret,-COST)
    # No scan segment may use a future exit beyond its boundary.
    t=trades(d,f,'pullback','2019-01-01',str(d.index[220].date()));assert len(t)==0
    # Adjustment factor must not scale actual execution prices or ATR distances.
    d2=d.copy()
    for col in ['open','high','low','close']:
        d2['raw_'+col]=d2[col];d2[col]*=.5
    f2=f.copy();f2['atr']*=.5
    t=trades(d2,f2,'pullback','2019-01-01','2020-01-01')
    assert t.iloc[0].entry==100 and t.iloc[0].atr==2 and np.isclose(t.iloc[0].ret,-COST)
    save('validation.json',dict(status='pass',checks=['Wilder flat RSI','ATR constant range','same-bar stop first','gap stop slippage','gap target order','entry-auction high excluded','segment endpoint purge','cost deduction','split-price execution and ATR unadjustment']))
if __name__=='__main__':
    checks()
    if '--fetch' in sys.argv:fetch()
    if '--discover' in sys.argv:discover()
    if '--test' in sys.argv:test()
