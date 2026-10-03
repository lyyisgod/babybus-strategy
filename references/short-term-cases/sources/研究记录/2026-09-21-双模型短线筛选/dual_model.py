"""Frozen short-horizon research model; no brokerage or order operations."""
from pathlib import Path
import json, sys, ssl, urllib.request, hashlib, importlib.util
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import certifi

ROOT=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('prior',ROOT.parent/'2026-09-17-高胜率操作/model.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
m.ROOT=ROOT;m.TODAY=pd.Timestamp('2026-09-21',tz=m.NY)
OFFLINE='--offline' in sys.argv
GROUPS={
'semi':'NVDA AVGO AMD MU SNDK MRVL ALAB TSM WDC STX CRDO LITE COHR AAOI AMAT LRCX KLAC ONTO CAMT TER SMTC VICR INTC CIEN',
'power_infra':'CLS FN VRT NBIS CRWV IREN BE GEV ETN CEG VST',
'growth':'RKLB ASTS PLTR IOVA ABCL CDNA TEAM AMPL RNG ATRC PBF',
'crypto':'MSTR COIN HOOD'}
STOCKS=list(dict.fromkeys(' '.join(GROUPS.values()).split()))
PROXIES='SPY QQQ IWM SMH XLU IGV XBI XLE JNK HYG TLT GLD'.split()+['^VIX','^TNX','DX-Y.NYB','BZ=F','NQ=F','ES=F']
SYMBOLS=list(dict.fromkeys(STOCKS+PROXIES))
def save(n,v):m.save(n,v)
def fetch(job):
    s,k=job
    url=(f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=5d&interval=5m&includePrePost=true' if k=='intraday' else f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=10y&interval=1d&events=div%2Csplits')
    path=ROOT/'raw'/f'{s}{"-intraday" if k=="intraday" else ""}.json'
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Research/1.0'})
        raw=urllib.request.urlopen(req,context=ssl.create_default_context(cafile=certifi.where()),timeout=25).read()
        assert json.loads(raw)['chart']['result'][0]['timestamp']
        path.write_bytes(raw)
        return dict(symbol=s,kind=k,url=url,fetched_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest())
    except Exception as e:return dict(symbol=s,kind=k,url=url,error=str(e))
def quote(s,d):
    r=json.loads((ROOT/'raw'/f'{s}-intraday.json').read_text())['chart']['result'][0]
    b=pd.DataFrame(r['indicators']['quote'][0],index=pd.to_datetime(r['timestamp'],unit='s',utc=True).tz_convert(m.NY)).dropna(subset=['close'])
    now=pd.Timestamp.now(tz=m.NY);b=b[b.index<=now]
    today=b[b.index.date==m.TODAY.date()]
    last=b.iloc[-1];tm=b.index[-1];prev=d.raw_close.iloc[-1]
    return dict(symbol=s,price=last.close,quote_at=tm.isoformat(),quote_age_minutes=(now-tm).total_seconds()/60,quote_is_today=tm.date()==m.TODAY.date(),session='premarket' if tm.hour<9 or (tm.hour==9 and tm.minute<30) else 'regular_or_post',change_pct=100*(last.close/prev-1),previous_close=prev,pre_volume=float(today.volume.sum()) if len(today) and today.volume.sum()>0 else None,volume_note='Zero/missing extended-session volume is unavailable, not actual zero trading.',regular_price=r['meta'].get('regularMarketPrice'),regular_time=datetime.fromtimestamp(r['meta']['regularMarketTime'],m.NY).isoformat())
def factors(s,d,ds):
    c=d.close;f=m.factors(d);x=pd.DataFrame(index=d.index)
    for n in [1,3,5,20,60]:x[f'ret{n}']=c.pct_change(n)
    x['rsi2']=f.rsi2/100;x['atr_pct']=f.atr/c
    for n in [20,50,200]:x[f'ema{n}_dist']=c/f[f'ema{n}']-1
    x['rv20']=c.pct_change().rolling(20).std()
    x['volume_ratio']=d.volume/d.volume.shift().rolling(20).mean()
    x['range_pos']=(c-d.low)/(d.high-d.low).replace(0,np.nan)
    x['drawdown60']=c/c.rolling(60).max()-1
    for b,n in [('QQQ',1),('QQQ',5),('QQQ',20),('SMH',5),('JNK',20),('TLT',20)]:x[f'{b}_r{n}']=ds[b].close.pct_change(n).reindex(x.index)
    x['rs20']=x.ret20-x.QQQ_r20
    return x.replace([np.inf,-np.inf],np.nan)
def logistic(x,y):
    beta=np.zeros(x.shape[1]);beta[0]=np.log((y.mean()+1e-6)/(1-y.mean()+1e-6))
    penalty=np.eye(x.shape[1])*.01;penalty[0,0]=0
    for _ in range(30):
        p=1/(1+np.exp(-np.clip(x@beta,-30,30)))
        grad=x.T@(p-y)/len(y)+penalty@beta
        h=(x.T*(p*(1-p)))@x/len(y)+penalty+np.eye(x.shape[1])*1e-8
        step=np.linalg.solve(h,grad);beta-=step
        if np.max(np.abs(step))<1e-7:break
    return beta
def main():
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),stocks=STOCKS,proxies=PROXIES,train='2016-2022',validation='2023-2024',evaluation='2025-2026-09-18',execution='Signal at completed close, next regular open; next-day target is exit at second session close; three-session target is third close.',targets=['net close return >=5% by second-session close','net close return >=8% by third-session close','net three-session return'],cost_round_trip=.002,model='Train-standardized features, fixed ridge 0.01; pooled logistic classifiers and linear ridge; no ticker identifiers; no hyperparameter search.',selection='Rank current stocks by model p(next-day +5%), then expected three-session return. Buy only if p>=0.5, predicted net return>=2%, fresh current quote, opening-gap proxy <=2%, Babybus credit ON and stock trigger; historical selected trades must have positive date-block mean lower bound. Daily top-1 evaluation uses nonoverlapping three-session calendar slots.',limitations=['Current survivor list; not all-market or point-in-time membership','Five movers added before modeling: INTC CIEN MSTR COIN HOOD; selection bias disclosed','Historical series previously inspected; evaluation is temporal out-of-training, not an untouched institutional validation','No premarket execution history, no point-in-time news/EPS/options features. Current quote only execution filter; not fed into model trained on completed daily bars.'])
    if not OFFLINE:
        save('dual-protocol.json',protocol)
        jobs=[(s,'daily') for s in SYMBOLS if not (ROOT/'raw'/f'{s}.json').exists()]+[(s,'intraday') for s in SYMBOLS]
        with ThreadPoolExecutor(max_workers=6) as ex:manifest=list(ex.map(fetch,jobs))
        save('dual-manifest.json',manifest)
        for series in ['BAMLH0A0HYM2','DFII10']:
            url=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2026-06-01&coed=2026-09-21'
            try:
                data=urllib.request.urlopen(url,context=ssl.create_default_context(cafile=certifi.where()),timeout=25).read();(ROOT/'raw'/f'{series}.csv').write_bytes(data)
                save(f'{series}-source.json',dict(url=url,fetched_at=datetime.now(timezone.utc).isoformat()))
            except Exception as e:save(f'{series}-source.json',dict(url=url,error=str(e)))
    ds={};errors={};quotes=[]
    for s in SYMBOLS:
        try:ds[s]=m.load(s)[0];quotes.append(quote(s,ds[s]))
        except Exception as e:errors[s]=str(e)
    save('quotes.json',quotes);pd.DataFrame(quotes).to_csv(ROOT/'quotes.csv',index=False)
    save('data-errors.json',errors)
    panels=[];live=[];tech=[]
    for s in STOCKS:
        if s not in ds:continue
        d=ds[s];x=factors(s,d,ds);cols=list(x.columns)
        # Outcomes use split-adjusted executable OHLC, excluding cash dividends.
        x['r2']=d.raw_close.shift(-2)/d.raw_open.shift(-1)-1-.002
        x['r3']=d.raw_close.shift(-3)/d.raw_open.shift(-1)-1-.002
        x['exit_at']=pd.Series(d.index,index=d.index).shift(-3)
        x['signal_at']=d.index;x['symbol']=s
        live.append(x.iloc[-1]);panels.append(x.dropna(subset=cols+['r2','r3','exit_at']))
        f=m.factors(d);w=d.close.resample('W-FRI').last();w=w[w.index<=d.index[-1]]
        mid=w.iloc[-20:].mean();sd=w.iloc[-20:].std(ddof=0);lo=mid-2*sd
        p=d.close.iloc[-1];z=(p-d.close.tail(20).mean())/d.close.tail(20).std(ddof=0)
        trigger=bool((d.close.iloc[-2]<=lo and p>lo) or (lo<=p<=mid and p>d.high.iloc[-2]))
        ratio=p/d.raw_close.iloc[-1]
        tech.append(dict(symbol=s,asof=str(d.index[-1].date()),rsi2=f.rsi2.iloc[-1],atr=f.atr.iloc[-1]/ratio,weekly_lower=lo/ratio,weekly_mid=mid/ratio,weekly_upper=(mid+2*sd)/ratio,bb_pos=(p-mid)/(2*sd),daily_z20=z,baby_price_trigger=trigger,baby_no_chase=bool((p-mid)/(2*sd)<.8 and z<2),original_rules=[r for r in m.RULES if f[r].iloc[-1]],original_loc_ceiling=(p+.5*f.atr.iloc[-1])/ratio,ret20_pct=100*x.ret20.iloc[-1],rs20_qqq_pp=100*x.rs20.iloc[-1],volume_ratio=x.volume_ratio.iloc[-1]))
    panel=pd.concat(panels);current=pd.DataFrame(live).dropna(subset=cols)
    train=panel[(panel.signal_at.dt.year<=2022)&(panel.exit_at.dt.year<=2022)]
    mu=train[cols].mean().to_numpy();sd=train[cols].std().to_numpy().copy();sd[sd==0]=1
    def design(a):return np.c_[np.ones(len(a)),np.clip((a[cols].to_numpy()-mu)/sd,-8,8)]
    X=design(train);Y=[(train.r2>=.05).to_numpy(float),(train.r3>=.08).to_numpy(float)]
    bs=[logistic(X,y) for y in Y]
    reg=np.eye(X.shape[1])*.01;reg[0,0]=0
    br=np.linalg.solve(X.T@X/len(X)+reg,X.T@train.r3.to_numpy()/len(X))
    def predict(a):
        out=a.copy();xx=design(a)
        out['p_next5']=1/(1+np.exp(-np.clip(xx@bs[0],-30,30)))
        out['p_three8']=1/(1+np.exp(-np.clip(xx@bs[1],-30,30)))
        out['expected3']=xx@br;return out
    metrics={};evals=[]
    for name,lo,hi in [('validation',2023,2024),('evaluation',2025,2026)]:
        a=predict(panel[(panel.signal_at.dt.year>=lo)&(panel.exit_at.dt.year<=hi)])
        pred=a.p_next5.to_numpy();yy=(a.r2>=.05).to_numpy(float)
        top=a.sort_values(['signal_at','p_next5','expected3'],ascending=[True,False,False]).groupby('signal_at',as_index=False).head(1).sort_values('signal_at')
        chosen=[];busy=None
        for _,r in top.iterrows():
            if busy is None or r.signal_at>=busy:chosen.append(r);busy=r.exit_at
        t=pd.DataFrame(chosen)
        gate=t[(t.p_next5>=.5)&(t.expected3>=.02)]
        benchmark=a.groupby('signal_at').r3.mean().reindex(t.signal_at).to_numpy()
        difference=t.r3.to_numpy()-benchmark
        metrics[name]=dict(rows=len(a),dates=a.signal_at.nunique(),base_rate=yy.mean(),brier=((pred-yy)**2).mean(),constant_train_brier=((Y[0].mean()-yy)**2).mean(),top1_nonoverlap_n=len(t),top1_mean_net3=t.r3.mean(),same_dates_pool_equalweight_mean3=benchmark.mean(),top1_excess_mean3=difference.mean(),top1_excess_block95=m.block_mean_ci(difference),top1_win3=(t.r3>0).mean(),top1_next5_rate=(t.r2>=.05).mean(),top1_three8_rate=(t.r3>=.08).mean(),top1_mean_block95=m.block_mean_ci(t.r3.to_numpy()),threshold_qualified_n=len(gate),threshold_mean_net3=gate.r3.mean(),calibration=[dict(bin=f'{l:.1f}-{l+.1:.1f}',n=int(((pred>=l)&(pred<l+.1)).sum()),realized=float(yy[(pred>=l)&(pred<l+.1)].mean()) if ((pred>=l)&(pred<l+.1)).any() else None) for l in np.arange(0,1,.1)])
        t[['symbol','signal_at','exit_at','p_next5','p_three8','expected3','r2','r3']].to_csv(ROOT/f'{name}-top1.csv',index=False)
        assert (a.exit_at>a.signal_at).all()
    cur=predict(current);q=pd.DataFrame(quotes);t=pd.DataFrame(tech)
    ranked=cur[['symbol','p_next5','p_three8','expected3']].merge(t,on='symbol').merge(q,on='symbol').sort_values(['p_next5','expected3'],ascending=False)
    ranked.to_csv(ROOT/'ranking.csv',index=False)
    save('model-metrics.json',metrics)
    save('model-coefficients.json',dict(features=cols,train_mean=mu.tolist(),train_sd=sd.tolist(),logistic=[b.tolist() for b in bs],linear=br.tolist(),train_rows=len(train)))
    # Babybus current gate: completed JNK months and weeks, no fabricated macro vintages.
    j=ds['JNK'].close;weeks=j.resample('W-FRI').last();weeks=weeks[weeks.index<=j.index[-1]]
    months=j.resample('ME').last();months=months[months.index<j.index[-1].replace(day=1)]
    bear=months.tail(3).mean()<months.tail(10).mean();breakdown=weeks.iloc[-1]<weeks.iloc[-9:-1].min()*.995
    macro={}
    for series in ['BAMLH0A0HYM2','DFII10']:
        try:
            a=pd.read_csv(ROOT/'raw'/f'{series}.csv');a.iloc[:,1]=pd.to_numeric(a.iloc[:,1],errors='coerce');a=a.dropna();a.index=pd.to_datetime(a.iloc[:,0]);a=a[a.index<=pd.Timestamp('2026-09-18')]
            last=a.index[-1];ix=ds['QQQ'].index.tz_localize(None);past=ix[ix<=last][-21];prior=a[a.index<=past].iloc[-1,1]
            macro[series]=dict(date=str(last.date()),value=float(a.iloc[-1,1]),change20_bp=100*(float(a.iloc[-1,1])-float(prior)),calendar_age_days=(pd.Timestamp('2026-09-21')-last).days)
        except Exception as e:macro[series]=dict(error=str(e))
    oas=macro['BAMLH0A0HYM2'].get('change20_bp');jr=j.iloc[-1]/j.iloc[-21]-1
    off=bool(bear or breakdown or (oas is not None and oas>=50))
    # 3-day ON confirmation cannot be proven by a single contemporary OAS snapshot.
    state='OFF' if off else ('UNKNOWN' if oas is None else ('NOT_ON' if jr<=0 or oas>0 else 'ON_CANDIDATE_UNCONFIRMED'))
    save('babybus.json',dict(asof='2026-09-18 close',state=state,jnk_ret20_pct=100*jr,completed_month_fast3=months.tail(3).mean(),completed_month_slow10=months.tail(10).mean(),bear_month=bool(bear),weekly_breakdown=bool(breakdown),macro=macro,three_day_ON_confirmed=False,qualifying_price_triggers=ranked.loc[ranked.baby_price_trigger&ranked.baby_no_chase,'symbol'].tolist(),new_buy_permitted=False,note='Current gate and price triggers only; not a full account-level Babybus backtest. Known OFF dominates missing inputs.'))
    # Deterministic checks of time labels and numerical fit mechanics.
    toy=pd.Series([100.,110.,120.,130.,140.]);assert np.isclose((toy.shift(-2)/toy.shift(-1)-1-.002).iloc[0],120/110-1-.002)
    z=np.c_[np.ones(100),np.zeros(100)];assert abs((1/(1+np.exp(-logistic(z,np.r_[np.ones(20),np.zeros(80)])[0])))-.2)<1e-6
    assert train.exit_at.max()<pd.Timestamp('2023-01-01',tz=m.NY)
    assert all(d.index.max()<m.TODAY for d in ds.values())
    save('dual-validation.json',dict(status='pass',checks=['next-open future-close label arithmetic','logistic intercept known 20% prevalence','training outcomes end before 2023','incomplete current-session daily bars excluded','chronological forward outcomes'],not_validated=['premarket execution','historical point-in-time universe and macro vintages','causal influence of news','prospective live profitability']))
    print(ranked[['symbol','price','change_pct','p_next5','p_three8','expected3','baby_price_trigger','original_rules']].head(15).to_string(index=False));print(json.dumps(m.clean(metrics),indent=2));print((ROOT/'babybus.json').read_text())
if __name__=='__main__':main()
