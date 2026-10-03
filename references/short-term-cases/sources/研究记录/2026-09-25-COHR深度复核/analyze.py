"""Frozen, descriptive single-stock execution audit. No fitted probabilities."""
from pathlib import Path
import importlib.util, json, hashlib
from statistics import NormalDist
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prior', R.parent/'2026-09-17-高胜率操作/model.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
P = json.loads((R/'protocol.json').read_text())
NY = 'America/New_York'

def save(name, obj):
    (R/name).write_text(json.dumps(prior.clean(obj), ensure_ascii=False, indent=2, allow_nan=False))

def read(symbol, interval='1d'):
    x = json.loads((R/'raw'/f'{symbol}-{interval}.json').read_text())['chart']['result'][0]
    ix = pd.to_datetime(x['timestamp'], unit='s', utc=True).tz_convert(NY)
    d = pd.DataFrame(x['indicators']['quote'][0], index=ix).dropna(subset=['open','high','low','close'])
    if interval == '1d':
        d.index = d.index.normalize()
        d = d[d.index <= pd.Timestamp('2026-09-25', tz=NY)]
    return d, x['meta']

def risk_exit(d, e, entry, floor, first_high, intraday_entry):
    target = entry * 1.202
    for j in range(e if intraday_entry else e+1, e+11):
        row = d.iloc[j]
        # Entry-day target excluded conservatively, but closing invalidation is observable.
        if j > e and row.open >= target:
            return j, float(row.open), 'gap_target'
        if j > e and row.high >= target:
            return j, target, 'target'
        if j == e+10:
            return j, float(row.close), 'time10'
        if row.close < floor:
            return j+1, float(d.open.iloc[j+1]), 'close_stop_next_open'
        if j == e+3 and row.close <= first_high:
            return j+1, float(d.open.iloc[j+1]), 'time_failure_next_open'
    raise AssertionError('missing exit')

def stats(t):
    if not len(t):
        return {'n':0}
    a = t.net.to_numpy(); n=len(a); k=int((a>0).sum())
    rng=np.random.default_rng(20260925); boot=[]
    ex=t.excess_vs_hold10.to_numpy()
    for _ in range(4000):
        starts=rng.integers(0,n,int(np.ceil(n/3)))
        ix=((starts[:,None]+np.arange(3))%n).ravel()[:n]
        boot.append([a[ix].mean(),ex[ix].mean()])
    ci=np.quantile(boot,[.025,.975],axis=0)
    eq=np.r_[1.,np.cumprod(1+a)]
    return {'n':n,'wins':k,'win_rate':k/n,'wilson95':prior.wilson(k,n),
            'six_comparison_wilson':prior.wilson(k,n,NormalDist().inv_cdf(1-.05/12)),
            'mean':a.mean(),'median':np.median(a),'mean_block95':ci[:,0].tolist(),
            'worst':a.min(),'mean_win':a[a>0].mean() if k else None,
            'mean_loss':a[a<=0].mean() if k<n else None,
            'mean_cost_40bp':a.mean()-.002,'mean_cost_50bp':a.mean()-.003,
            'positive_cost_40bp':float((a>.002).mean()),
            'net20_realized_count':int((a>=.20-1e-10).sum()),
            'net20_touch10_count':int(t.touch20.sum()),
            'median_mae10':t.mae10.median(),'worst_mae10':t.mae10.min(),
            'closed_trade_drawdown':(eq/np.maximum.accumulate(eq)-1).min(),
            'mean_excess_hold10':ex.mean(),'excess_block95':ci[:,1].tolist(),
            'mean_hold10_excess_SMH':t.hold10_excess_smh.mean(),
            'reasons':t.reason.value_counts().to_dict()}

def main():
    d,meta=read('COHR'); f=prior.factors(d); smh,_=read('SMH')
    assert d.index.is_unique and d.index.is_monotonic_increasing
    assert ((d.high>=d[['open','close']].max(axis=1)-1e-5)&(d.low<=d[['open','close']].min(axis=1)+1e-5)).all()
    rows=[]; attempts=[]; anchors=[]
    for segment,(start,end) in P['segments'].items():
        ids=np.flatnonzero((d.index>=pd.Timestamp(start,tz=NY))&(d.index<=pd.Timestamp(end,tz=NY)))
        last=ids[-1];busy=-1
        for i in ids:
            if i<200 or i<=busy or i+21>last or not f.pullback.iloc[i]:
                continue
            first=d.iloc[i+1]; sig=d.iloc[i]; atr=float(f.atr.iloc[i]); floor=float(f.sma200.iloc[i])
            cap=float(sig.close+.5*atr)
            if not (first.close>sig.close and first.close>floor and first.close<=cap):
                continue
            busy=i+21
            anchors.append({'segment':segment,'signal':str(d.index[i].date()),'index':int(i)})
            for method in P['entries']:
                e=None; price=None
                if method=='auction':
                    e=i+1;price=float(first.close)
                elif method=='delayed_open' and floor<d.open.iloc[i+2]<=cap:
                    e=i+2;price=float(d.open.iloc[e])
                elif method=='confirmed':
                    for c in range(i+2,i+5):
                        if d.close.iloc[c]>first.high:
                            if floor<d.open.iloc[c+1]<=first.close+.5*atr:
                                e=c+1;price=float(d.open.iloc[e])
                            break
                attempts.append({'segment':segment,'signal':str(d.index[i].date()),'entry_method':method,'filled':e is not None})
                if e is None:continue
                bench=float(d.close.iloc[e+10]/price-1-.002)
                bars=d.iloc[e+1:e+11]
                smhret=float(smh.loc[d.index[e+10],'close']/smh.loc[d.index[e],'close' if method=='auction' else 'open']-1-.002)
                for exit_method in P['exits']:
                    ex,px,reason=(e+10,float(d.close.iloc[e+10]),'hold10') if exit_method=='hold10' else risk_exit(d,e,price,floor,float(first.high),method!='auction')
                    net=px/price-1-.002
                    rows.append({'segment':segment,'signal':str(d.index[i].date()),'signal_index':int(i),'entry_method':method,'exit_method':exit_method,'entry_date':str(d.index[e].date()),'exit_date':str(d.index[ex].date()),'entry_index':e,'exit_index':ex,'entry':price,'exit':px,'floor':floor,'net':net,'hold10':bench,'excess_vs_hold10':net-bench,'hold10_excess_smh':bench-smhret,'touch20':bool(bars.high.max()/price-1-.002>=.20),'mae10':float(bars.low.min()/price-1),'reason':reason})
    t=pd.DataFrame(rows);t.to_csv(R/'trades.csv',index=False)
    save('attempts.json',attempts);save('anchors.json',anchors)
    summaries={}
    for entry in P['entries']:
        summaries[entry]={}
        for ex in P['exits']:
            sub=t[(t.entry_method==entry)&(t.exit_method==ex)]
            summaries[entry][ex]={'all':stats(sub)}
            for segment in P['segments']:
                summaries[entry][ex][segment]=stats(sub[sub.segment==segment])
            summaries[entry][ex]['by_year']={str(y):stats(g) for y,g in sub.groupby(sub.entry_date.str[:4])}
    save('results.json',summaries)
    # Matched historical entry comparison; no selection on realized outcomes.
    matched={}
    for seg in ['all',*P['segments']]:
        a=t[t.exit_method=='hold10']
        if seg!='all':a=a[a.segment==seg]
        pivot=a.pivot(index='signal',columns='entry_method',values='net')
        matched[seg]={}
        for method in ['delayed_open','confirmed']:
            pair=pivot[['auction',method]].dropna();delta=pair[method]-pair.auction
            matched[seg][method]={'n':len(pair),'mean_difference':delta.mean(),'outperform_count':int((delta>0).sum())}
    save('matched.json',matched)
    # Current completed-day technicals, no today's partial daily inputs.
    b,_=read('COHR','5m');today=b[(b.index.date==d.index[-1].date())&(b.index.hour<16)]
    vwap=float((((today.high+today.low+today.close)/3)*today.volume).sum()/today.volume.sum())
    q,_=read('COHR','1m');post=q[(q.index.date==d.index[-1].date())&(q.index.hour>=16)]
    technical=[]
    for s in ['COHR','SMH','QQQ','LITE','FN']:
        sd,sm=read(s);sf=prior.factors(sd)
        technical.append({'symbol':s,'day':str(sd.index[-1].date()),'close':float(sd.close.iloc[-1]),'change_pct':float((sd.close.iloc[-1]/sd.close.iloc[-2]-1)*100),'return20_pct':float((sd.close.iloc[-1]/sd.close.iloc[-21]-1)*100),'volume_ratio20':float(sd.volume.iloc[-1]/sd.volume.iloc[-21:-1].mean()),'rsi2':sf.rsi2.iloc[-1],'sma200':sf.sma200.iloc[-1],'ema20':sf.ema20.iloc[-1],'atr14':sf.atr.iloc[-1],'new_pullback_signal':sf.pullback.iloc[-1],'quote_at':pd.Timestamp(sm['regularMarketTime'],unit='s',tz='UTC').tz_convert(NY).isoformat()})
    current={'retrieved_at':json.loads((R/'manifest.json').read_text())[0]['retrieved_at'],'technical':technical,'vwap_proxy':vwap,'day_high':float(d.high.iloc[-1]),'day_low':float(d.low.iloc[-1]),'day_open':float(d.open.iloc[-1]),'close_location':float((d.close.iloc[-1]-d.low.iloc[-1])/(d.high.iloc[-1]-d.low.iloc[-1])),'previous_signal_active':bool(f.pullback.iloc[-2]),'original_entry_cap':float(d.close.iloc[-2]+.5*f.atr.iloc[-2]),'confirmation_entry_cap':float(d.close.iloc[-1]+.5*f.atr.iloc[-2]),'postmarket_reference':None if not len(post) else {'time':post.index[-1].isoformat(),'price':float(post.close.iloc[-1]),'volume':float(post.volume.iloc[-1]),'bid_ask':None},'atr_fraction':float(f.atr.iloc[-1]/d.close.iloc[-1]),'structural_stop_fraction_at296':float(1-f.sma200.iloc[-2]/296),'old_stop_distance_atr':float((296-f.sma200.iloc[-2])/f.atr.iloc[-2]),'price_to_consensus_FY2027_EPS':float(d.close.iloc[-1]/9.42),'valuation_sensitivity':{str(eps):{str(pe):eps*pe for pe in [25,30,35,40]} for eps in [8.26,9.42,10.26]}}
    save('current.json',current)
    # Feature causality at multiple historical prefixes.
    for cut in [500,1200,len(d)-1]:
        pf=prior.factors(d.iloc[:cut])
        assert np.allclose(pf.iloc[-1][['rsi2','atr','sma200']].astype(float),f.iloc[cut-1][['rsi2','atr','sma200']].astype(float),equal_nan=True)
    # Close invalidation must exit next open, not magically at threshold.
    toy=pd.DataFrame({'open':[100.]*12,'high':[101.]*12,'low':[99.]*12,'close':[100.]*12})
    toy.loc[1,'close']=94.;toy.loc[2,'open']=90.
    assert risk_exit(toy,0,100,95,100,False)==(2,90.,'close_stop_next_open')
    toy.loc[1,'close']=100.;toy.loc[2,'open']=100.;toy.loc[1,'low']=90.
    assert risk_exit(toy,0,100,95,100,False)==(4,100.,'time_failure_next_open')
    toy.loc[1,'high']=125.
    assert risk_exit(toy,0,100,95,100,False)==(1,120.19999999999999,'target')
    for (_,_,_),g in t.groupby(['segment','entry_method','exit_method']):
        assert (g.entry_index.iloc[1:].to_numpy()>g.entry_index.iloc[:-1].to_numpy()+10).all()
    for item in json.loads((R/'manifest.json').read_text()):
        assert hashlib.sha256((R/'raw'/item['file']).read_bytes()).hexdigest()==item['sha256']
    save('validation.json',{'feature_prefix_consistency':True,'daily_ohlc_consistency':True,'close_stop_next_open_gap_case':True,'intraday_breach_not_close_stop':True,'target_arithmetic':True,'nonoverlap':True,'hashes':True,'data_last_complete_day':str(d.index[-1].date()),'current_probability_validated':False,'afterhours_execution_validated':False})
    print(json.dumps(prior.clean(current),ensure_ascii=False,indent=2))
    for en in summaries:
        for ex in summaries[en]:
            print(en,ex,{k:{z:v.get(z) for z in ['n','wins','mean','worst','net20_realized_count','net20_touch10_count','mean_excess_hold10']} for k,v in summaries[en][ex].items() if k!='by_year'})

if __name__=='__main__':main()
