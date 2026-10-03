"""One-session movement research. No executable after-hours backtest or trading API."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'protocol.json').read_text())
NY=ZoneInfo('America/New_York');DAY=pd.Timestamp(P['date'],tz=NY)
STOCKS=P['universe'].split();PROXIES=P['proxies'].split();TARGETS=['hit5','hit10','close3','down3'];COST=.002
FITS=[]

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(datetime,pd.Timestamp)):return x.isoformat()
    return x
def save(name,x):(ROOT/name).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False))
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-35,35)))
def fitlog(x,y,penalty=.01):
    b=np.zeros(x.shape[1]);b[0]=np.log((y.mean()+1e-7)/(1-y.mean()+1e-7))
    reg=np.eye(x.shape[1])*penalty;reg[0,0]=0
    def loss(beta):
        z=x@beta
        return np.mean(np.logaddexp(0,z)-y*z)+.5*beta@reg@beta
    converged=False
    for iteration in range(100):
        p=sigmoid(x@b);g=x.T@(p-y)/len(y)+reg@b
        if np.max(np.abs(g))<1e-7:converged=True;break
        h=(x.T*(p*(1-p)))@x/len(y)+reg+np.eye(x.shape[1])*1e-9
        step=np.linalg.solve(h,g);descent=g@step
        assert descent>0,'Newton direction must descend'
        old_loss=loss(b);rate=1.
        while rate>1e-10 and loss(b-rate*step)>old_loss-1e-4*rate*descent:rate*=.5
        assert rate>1e-10,'line search failed'
        b-=rate*step
    grad=x.T@(sigmoid(x@b)-y)/len(y)+reg@b
    converged=converged or np.max(np.abs(grad))<1e-6
    FITS.append(dict(rows=len(y),columns=x.shape[1],iterations=iteration+1,converged=converged,loss=loss(b),gradient_max=np.max(np.abs(grad)),max_coefficient=np.max(np.abs(b))))
    assert converged and np.isfinite(b).all(),'logistic fit did not converge'
    return b
def load(s):
    r=json.loads((ROOT/'raw'/f'{s}-1d.json').read_text())['chart']['result'][0]
    d=pd.DataFrame(r['indicators']['quote'][0],index=pd.to_datetime(r['timestamp'],unit='s',utc=True).tz_convert(NY).normalize())
    d['adj']=r['indicators']['adjclose'][0]['adjclose'];d=d[d.index<=DAY].dropna(subset=['open','high','low','close','volume','adj'])
    assert d.index.is_unique and d.index.is_monotonic_increasing
    assert ((d.high+1e-4>=d[['open','close']].max(axis=1))&(d.low-1e-4<=d[['open','close']].min(axis=1))).all(),s
    scale=d.adj/d.close
    for c in ['open','high','low','close']:d['raw_'+c]=d[c];d[c]*=scale
    return d,r
def wilder(s,n):
    a=s.to_numpy(float);v=np.full(len(a),np.nan)
    if len(a)>=n:
        v[n-1]=a[:n].mean()
        for i in range(n,len(a)):v[i]=(v[i-1]*(n-1)+a[i])/n
    return pd.Series(v,index=s.index)
def features(d,ds):
    c=d.close;r=c.pct_change();x=pd.DataFrame(index=d.index)
    for n in [1,3,5,20,60]:x[f'ret{n}']=c.pct_change(n)
    for n in [10,20,50,200]:x[f'ema{n}_dist']=c/c.ewm(span=n,adjust=False).mean()-1
    tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
    x['atr_pct']=wilder(tr,14)/c
    x['rv20']=r.rolling(20).std();x['rv60']=r.rolling(60).std()
    x['volume_ratio']=d.volume/d.volume.shift().rolling(20).mean()
    x['range_position']=(c-d.low)/(d.high-d.low).replace(0,np.nan)
    x['gap']=d.open/c.shift()-1;x['intraday']=c/d.open-1
    x['drawdown60']=c/c.rolling(60).max()-1
    x['high20_dist']=c/d.high.shift().rolling(20).max()-1
    g=wilder(c.diff().dropna().clip(lower=0),14);l=wilder(-c.diff().dropna().clip(upper=0),14)
    x['rsi14']=1-1/(1+g/l);x.loc[(l==0).reindex(x.index,fill_value=False),'rsi14']=1
    x.loc[((l==0)&(g==0)).reindex(x.index,fill_value=False),'rsi14']=.5
    for s,n in [('QQQ',1),('QQQ',5),('QQQ',20),('SMH',1),('SMH',5),('IWM',1),('JNK',20),('TLT',20),('BZ=F',5)]:
        series=ds[s].close.pct_change(n)
        if s=='BZ=F':series=series.shift(1) # Current futures daily session is still open at the stock decision time.
        x[f'{s}_r{n}']=series.reindex(x.index)
    x['VIX_level']=ds['^VIX'].close.reindex(x.index)/100
    x['VIX_r5']=ds['^VIX'].close.pct_change(5).reindex(x.index)
    x['rs20']=x.ret20-x.QQQ_r20
    q=ds['QQQ'].close.pct_change().reindex(x.index)
    x['beta60']=r.rolling(60).cov(q)/q.rolling(60).var()
    return x.replace([np.inf,-np.inf],np.nan)
def outcomes(d):
    o=d.raw_open.shift(-1);hi=d.raw_high.shift(-1);lo=d.raw_low.shift(-1);c=d.raw_close.shift(-1)
    z=pd.DataFrame(index=d.index)
    z['high_return']=hi/o-1;z['low_return']=lo/o-1;z['close_return']=c/o-1-COST
    z['priorclose_to_high']=hi/d.raw_close-1
    z['next_gap']=o/d.raw_close-1
    z['hit5']=(z.high_return>=.05).astype(float);z['hit10']=(z.high_return>=.10).astype(float)
    z['close3']=(z.close_return>=.03).astype(float);z['down3']=(z.low_return<=-.03).astype(float)
    z['toy_trade_return']=np.where(z.down3==1,-.03-COST,np.where(z.hit5==1,.05-COST,z.close_return))
    z['ambiguous_both']=(z.down3==1)&(z.hit5==1)
    z['exit_date']=pd.Series(d.index,index=d.index).shift(-1)
    return z
def quote(s,d):
    r=json.loads((ROOT/'raw'/f'{s}-5m.json').read_text())['chart']['result'][0]
    b=pd.DataFrame(r['indicators']['quote'][0],index=pd.to_datetime(r['timestamp'],unit='s',utc=True).tz_convert(NY)).dropna(subset=['close'])
    now=pd.Timestamp.now(tz=NY);b=b[b.index<=now];v=b.iloc[-1];t=b.index[-1]
    post=b[(b.index>=DAY+pd.Timedelta(hours=16))&(b.index<DAY+pd.Timedelta(hours=20))]
    validvol=post.volume.fillna(0).sum()
    return dict(symbol=s,last=float(v.close),last_at=t,age_minutes=(now-t).total_seconds()/60,
        session='after_hours' if t.date()==DAY.date() and t.hour>=16 else 'regular_or_stale',
        after_hours_change_pct=100*(v.close/d.raw_close.iloc[-1]-1),
        after_hours_volume=float(validvol) if validvol>0 else None,
        after_hours_high=post.high.max() if len(post) else None,after_hours_low=post.low.min() if len(post) else None,
        regular_quote_at=datetime.fromtimestamp(r['meta']['regularMarketTime'],NY),quote_note='Vendor latest observation, no executable bid/ask or overnight-session venue feed; after-hours bar volumes and extremes may include late regular-session batches and are not reliable liquidity/support measures')
def block_ci(a,seed=922):
    a=np.asarray(a,float);n=len(a)
    if n<10:return [None,None]
    rng=np.random.default_rng(seed);starts=rng.integers(0,n,size=(2500,int(np.ceil(n/5))))
    ix=((starts[:,:,None]+np.arange(5))%n).reshape(2500,-1)[:,:n]
    return np.quantile(a[ix].mean(axis=1),[.025,.975]).tolist()
def frequency_ci_by_date(a,target):
    g=a.groupby('date')[target].agg(['sum','count']);n=len(g)
    if n<10:return [None,None]
    rng=np.random.default_rng(922);starts=rng.integers(0,n,size=(1500,int(np.ceil(n/5))))
    ix=((starts[:,:,None]+np.arange(5))%n).reshape(1500,-1)[:,:n]
    vals=g['sum'].to_numpy()[ix].sum(axis=1)/g['count'].to_numpy()[ix].sum(axis=1)
    return np.quantile(vals,[.025,.975]).tolist()
def main():
    ds={};raw={};errors={}
    for s in STOCKS+PROXIES:
        try:ds[s],raw[s]=load(s)
        except Exception as e:errors[s]=str(e)
    assert all(s in ds for s in ['QQQ','SMH','IWM','JNK','TLT','BZ=F','^VIX'])
    panels=[];live=[];technical=[];quotes=[]
    for s in STOCKS:
        if s not in ds:continue
        d=ds[s];x=features(d,ds);cols=list(x.columns);x['symbol']=s;x['date']=x.index
        adv=(d.raw_close*d.volume).shift().rolling(20).mean()
        x['eligible']=(d.raw_close>=5)&(adv>=50e6)
        # Only mature numerical labels enter historical training/evaluation.
        p=x.join(outcomes(d));panels.append(p.iloc[63:].dropna(subset=cols+['high_return','exit_date']))
        live.append(x.iloc[-1]);quotes.append(quote(s,d))
        c=d.raw_close.iloc[-1];scale=d.close.iloc[-1]/c;f=x.iloc[-1];atr=f.atr_pct*c
        technical.append(dict(symbol=s,close=c,daily_pct=f.ret1*100,volume_ratio=f.volume_ratio,atr=atr,atr_pct=f.atr_pct*100,
            rsi=f.rsi14*100,ema20=c/(1+f.ema20_dist),ema50=c/(1+f.ema50_dist),rs20_qqq_pp=f.rs20*100,
            ret20_pct=f.ret20*100,range_position=f.range_position,high=d.raw_high.iloc[-1],low=d.raw_low.iloc[-1],
            prior_high20=d.raw_high.iloc[-21:-1].max(),adv20=adv.iloc[-1],history_sessions=len(d),
            high_close_pct=100*(d.raw_high.iloc[-1]/c-1),max_feature_abs=None))
    panel=pd.concat(panels);panel=panel[panel.eligible].copy()
    cur=pd.DataFrame(live).dropna(subset=cols).copy()
    tr=panel[panel.exit_date.dt.year<=2022];ca=panel[(panel.date.dt.year>=2023)&(panel.exit_date.dt.year<=2024)];te=panel[panel.date.dt.year>=2025].copy()
    mu=tr[cols].mean().to_numpy();sd=tr[cols].std().to_numpy().copy();sd[sd==0]=1
    def base(a):return np.clip((a[cols].to_numpy()-mu)/sd,-8,8)
    interactions=[('ret1','volume_ratio'),('range_position','volume_ratio'),('ret1','QQQ_r1'),('atr_pct','VIX_level'),('ret20','rs20')]
    def design(a,kind):
        z=base(a)
        if kind=='a':return np.c_[np.ones(len(a)),z]
        return np.c_[np.ones(len(a)),z,z*z,*[z[:,cols.index(l)]*z[:,cols.index(r)] for l,r in interactions]]
    coef={};cal={};metrics={}
    for kind in ['a','b']:
        tx,cx,ex,nx=[design(a,kind) for a in [tr,ca,te,cur]]
        for target in TARGETS:
            y=tr[target].to_numpy();cy=ca[target].to_numpy()
            b=fitlog(tx,y,.01 if kind=='a' else .03);cb=fitlog(np.c_[np.ones(len(cx)),cx@b],cy,.001)
            te[f'{kind}_{target}']=sigmoid(cb[0]+cb[1]*(ex@b));cur[f'{kind}_{target}']=sigmoid(cb[0]+cb[1]*(nx@b))
            coef[f'{kind}_{target}']=dict(coefficients=b,calibration=cb)
            print('fitted',kind,target,flush=True)
    for target in TARGETS:
        te[f'p_{target}']=(te[f'a_{target}']+te[f'b_{target}'])/2
        cur[f'p_{target}']=(cur[f'a_{target}']+cur[f'b_{target}'])/2
        y=te[target].to_numpy();p=te[f'p_{target}'].to_numpy();cp=ca[target].mean()
        metrics[target]=dict(event_rate=y.mean(),mean_prediction=p.mean(),brier=np.mean((p-y)**2),constant_calibration_brier=np.mean((cp-y)**2),
            bins=[dict(low=l,rows=int(((p>=l)&(p<l+.1)).sum()),prediction=p[(p>=l)&(p<l+.1)].mean(),actual=y[(p>=l)&(p<l+.1)].mean()) for l in np.arange(0,1,.1) if ((p>=l)&(p<l+.1)).sum()>=30])
    top=te.sort_values(['date','p_hit5'],ascending=[True,False]).groupby('date').head(1).sort_values('date')
    voltop=te.sort_values(['date','atr_pct'],ascending=[True,False]).groupby('date').head(1).sort_values('date')
    selection=dict(days=len(top),hit5_rate=top.hit5.mean(),hit5_dateblock95=block_ci(top.hit5),hit10_rate=top.hit10.mean(),
        close3_rate=top.close3.mean(),down3_rate=top.down3.mean(),net_close_mean=top.close_return.mean(),net_close_mean_dateblock95=block_ci(top.close_return),
        stop3_target5_mean=top.toy_trade_return.mean(),stop3_target5_dateblock95=block_ci(top.toy_trade_return),ambiguous_both_rate=top.ambiguous_both.mean(),
        fixed_highest_atr_hit5=voltop.hit5.mean(),same_date_equal_pool_hit5=te.groupby('date').hit5.mean().mean(),
        qualified50_count=int((top.p_hit5>=.5).sum()))
    top[['date','symbol','p_hit5','p_hit10','p_close3','p_down3','high_return','low_return','close_return','priorclose_to_high','next_gap','toy_trade_return','ambiguous_both']].to_csv(ROOT/'evaluation-top1.csv',index=False)
    empirical=[]
    for _,row in cur.iterrows():
        subset=te[(te.p_hit5-row.p_hit5).abs()<=.025]
        empirical.append(dict(symbol=row.symbol,evaluation_score_neighbors=len(subset),evaluation_score_dates=subset.date.nunique(),empirical_hit5=subset.hit5.mean(),
            empirical_hit5_dateblock95=frequency_ci_by_date(subset,'hit5'),model_disagreement=abs(row.a_hit5-row.b_hit5)))
    ranked=cur[['symbol','eligible']+[f'{k}_{t}' for k in ['a','b','p'] for t in TARGETS]].merge(pd.DataFrame(technical),on='symbol').merge(pd.DataFrame(quotes),on='symbol').merge(pd.DataFrame(empirical),on='symbol')
    ranked['extension_atr']=(ranked.close-ranked.ema20)/ranked.atr
    ranked=ranked.sort_values('p_hit5',ascending=False);ranked.to_csv(ROOT/'ranking.csv',index=False)
    ranked[ranked.eligible].to_csv(ROOT/'eligible-ranking.csv',index=False)
    save('ranking.json',ranked.to_dict('records'));save('errors.json',errors)
    save('metrics.json',dict(train_rows=len(tr),calibration_rows=len(ca),evaluation_rows=len(te),evaluation_dates=te.date.nunique(),features=len(cols),targets=metrics,selection=selection))
    save('coefficients.json',dict(features=cols,train_mean=mu,train_sd=sd,models=coef))
    save('fit-diagnostics.json',FITS)
    # Meaningful numerical/time validation rather than profitability certification.
    toy=pd.DataFrame({'raw_open':[100.,110.,80.],'raw_high':[101.,121.,88.],'raw_low':[99.,104.5,72.],'raw_close':[100.,115.5,84.]},index=pd.date_range('2020-01-01',periods=3,tz=NY))
    oo=outcomes(toy);assert np.isclose(oo.high_return.iloc[0],.10) and np.isclose(oo.close_return.iloc[0],.048)
    assert oo.toy_trade_return.iloc[0]==-.032 and oo.ambiguous_both.iloc[0]
    zz=np.c_[np.ones(100),np.zeros(100)];assert abs(sigmoid(fitlog(zz,np.r_[np.ones(20),np.zeros(80)])[0])-.2)<1e-6
    assert tr.exit_date.max()<pd.Timestamp('2023-01-01',tz=NY) and ca.exit_date.max()<pd.Timestamp('2025-01-01',tz=NY)
    d=ds['WDAY'];cut=300;fd=features(d,ds);pd.testing.assert_frame_equal(fd.iloc[:cut],features(d.iloc[:cut],ds))
    assert all(ds[s].index.max()==DAY for s in ranked.symbol)
    assert np.isfinite(cur[[f'p_{t}' for t in TARGETS]].to_numpy()).all()
    assert (ranked.p_hit10<=ranked.p_hit5+.01).all(),'material crossing of nested event probabilities'
    save('validation.json',dict(status='pass',checks=['target uses next regular open, not previous close','same-day target/stop ambiguity conservative','known intercept 20% prevalence','mature train/calibration label boundaries','feature prefix causality','complete Sep21 bars','finite probability and nested target sanity'],not_validated=['after-hours/overnight executions','point-in-time stock universe','causal effect of future announcements','prospective profitability']))
    print(ranked[['symbol','eligible','close','daily_pct','last','p_hit5','p_hit10','p_close3','p_down3','volume_ratio','extension_atr']].head(25).to_string(index=False))
    print(json.dumps(clean(selection),indent=2))
if __name__=='__main__':main()
