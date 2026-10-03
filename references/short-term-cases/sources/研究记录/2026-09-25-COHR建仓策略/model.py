"""Frozen daily-bar execution comparison; descriptive research, not calibrated odds."""
from pathlib import Path
from datetime import datetime,timezone
import json, hashlib, importlib.util
import numpy as np
import pandas as pd
from statistics import NormalDist

R=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('old',R.parent/'2026-09-17-高胜率操作/model.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
NY='America/New_York';TODAY=pd.Timestamp('2026-09-25',tz=NY)
def save(n,x):(R/n).write_text(json.dumps(old.clean(x),ensure_ascii=False,indent=2,allow_nan=False))
def read(fn):
    x=json.loads((R/'raw'/fn).read_text())['chart']['result'][0]
    d=pd.DataFrame(x['indicators']['quote'][0],index=pd.to_datetime(x['timestamp'],unit='s',utc=True).tz_convert(NY)).dropna(subset=['open','high','low','close'])
    return d,x['meta']
def outcome(b,entry,atr):
    stop,target=entry-2*atr,entry+2*atr
    for j,row in enumerate(b.itertuples()):
        if row.open<=stop:return j,row.open,'gap_stop'
        if row.open>=target:return j,row.open,'gap_target'
        if row.low<=stop:return j,stop,'stop'
        if row.high>=target:return j,target,'target'
    return len(b)-1,float(b.close.iloc[-1]),'time'
def trades(d,f,variant,start,end):
    ids=np.flatnonzero((d.index>=pd.Timestamp(start,tz=NY))&(d.index<=pd.Timestamp(end,tz=NY))); last=ids[-1];busy=-1;rows=[];attempts=[]
    for i in ids:
        if i<200 or i<=busy or not f.pullback.iloc[i]:continue
        e=i+1
        if e+20>last:continue
        sig=d.iloc[i];n=d.iloc[e];atr=f.atr.iloc[i];cap=sig.close+.5*atr;entry=None;reason='price_not_eligible'
        if variant=='open' and n.open<=cap:entry=n.open
        if variant=='auction' and n.close<=cap:entry=n.close
        if variant=='pullback':
            if n.open<=sig.close:entry=n.open
            elif n.low<sig.close:entry=sig.close
        if variant=='breakout':
            trigger=sig.high
            if trigger<=cap and n.open<=cap:
                if n.open>=trigger:entry=n.open
                elif n.high>trigger:entry=trigger
        attempts.append(dict(signal=str(d.index[i].date()),variant=variant,filled=entry is not None))
        if entry is None:continue
        if variant!='auction' and n.low<=entry-2*atr:
            ex=e;px=entry-2*atr;reason='entry_day_conservative_stop'
        else:
            j,px,reason=outcome(d.iloc[e+1:e+21],entry,atr);ex=e+1+j
        net=px/entry-1-.002;ten=d.close.iloc[e+10]/entry-1-.002
        bench=d.close.iloc[e+20]/entry-1-.002
        rows.append(dict(variant=variant,signal=str(d.index[i].date()),entry_date=str(d.index[e].date()),exit_date=str(d.index[ex].date()),entry_index=e,exit_index=ex,entry=entry,atr=atr,exit=px,reason=reason,net=net,net_double_cost=net-.002,net10=ten,ten_net_atleast10=ten>=.1,benchmark20=bench,excess_vs_hold20=net-bench))
        busy=e+20 # blocks overlapping 10-day/20-day labels as well as actual positions
    return pd.DataFrame(rows),attempts
def summary(t):
    if not len(t):return dict(n=0)
    a=t.net.to_numpy();wins=int((a>0).sum());eq=np.r_[1,np.cumprod(1+a)];rng=np.random.default_rng(20260925)
    boot=[];excess=t.excess_vs_hold20.to_numpy()
    for _ in range(4000):
        starts=rng.integers(0,len(a),int(np.ceil(len(a)/3)));ix=((starts[:,None]+np.arange(3))%len(a)).ravel()[:len(a)];boot.append([a[ix].mean(),excess[ix].mean()])
    ci=np.quantile(np.array(boot),[.025,.975],axis=0)
    return dict(n=len(a),wins=wins,win=wins/len(a),wilson95=old.wilson(wins,len(a)),four_variant_adjusted_wilson=old.wilson(wins,len(a),NormalDist().inv_cdf(1-.05/8)),mean=a.mean(),median=float(np.median(a)),mean_block95=ci[:,0].tolist(),excess_block95=ci[:,1].tolist(),worst=float(a.min()),max_closed_trade_drawdown=float((eq/np.maximum.accumulate(eq)-1).min()),mean_double_cost=t.net_double_cost.mean(),win_double_cost=float((t.net_double_cost>0).mean()),mean_adverse_extra_10bp=t.net_double_cost.mean()-.001,ten_net_atleast10=float(t.ten_net_atleast10.mean()),mean_net10=t.net10.mean(),mean_hold20=t.benchmark20.mean(),excess_vs_hold20=t.excess_vs_hold20.mean(),reasons=t.reason.value_counts().to_dict(),bootstrap_caution='3 sequential nonoverlapping trade blocks; descriptive, not untouched audit; extra10bp is cost stress, not full fill-resimulation')
def main():
    d,_=read('COHR-1d.json');d.index=d.index.normalize();d=d[d.index<TODAY];f=old.factors(d)
    proto=json.loads((R/'protocol.json').read_text());alltr=[];stats={};attempts=[]
    for variant in proto['entry_variants']:
        stats[variant]={}
        for name,start,end in [('development','2016-01-01','2022-12-31'),('validation','2023-01-01','2024-12-31'),('descriptive_recent','2025-01-01','2026-09-24')]:
            t,att=trades(d,f,variant,start,end);attempts+=att
            if len(t):t['segment']=name;alltr.append(t)
            stats[variant][name]=summary(t)
    alltr=pd.concat(alltr,ignore_index=True);alltr.to_csv(R/'trades.csv',index=False);save('attempts.json',attempts);save('results.json',stats)
    # Cutoff invariance, barrier ordering and known arithmetic checks.
    prefix=old.factors(d.iloc[:-100]);assert np.allclose(prefix.iloc[-1][['rsi2','atr','sma200']].astype(float),f.iloc[-101][['rsi2','atr','sma200']].astype(float),equal_nan=True)
    toy=pd.DataFrame({'open':[100.],'high':[106.],'low':[94.],'close':[101.]});assert outcome(toy,100,2)==(0,96,'stop')
    toy['open']=94;assert outcome(toy,100,2)==(0,94.,'gap_stop')
    for _,t in alltr.groupby(['variant','segment']):assert (t.entry_index.iloc[1:].to_numpy()>t.entry_index.iloc[:-1].to_numpy()+20).all()
    for x in json.loads((R/'manifest.json').read_text()):assert hashlib.sha256((R/'raw'/x['file']).read_bytes()).hexdigest()==x['sha256']
    save('validation.json',dict(causal_prefix_pass=True,barrier_double_touch_stop_first=True,gap_stop_arithmetic=True,nonoverlap_20sessions=True,all_source_hashes=True,daily_cutoff=str(d.index[-1].date()),signal_count=int(f.pullback.sum()),not_validated=['current win probability','intraday fills','independent untouched audit']))
    # Fresh intraday context. Only complete, aligned 5-minute bars used for VWAP proxy and volume comparisons.
    snap=[];now=pd.Timestamp.now(tz=NY)
    for s in 'COHR LITE FN CIEN SMH QQQ SPY JNK'.split():
        b,meta=read(s+'-5m.json');today=b[b.index.date==TODAY.date()];done=today[(today.index.minute%5==0)&(today.index+pd.Timedelta(minutes=5)<=now)];previous=b[b.index.date<TODAY.date()]
        dp=R/'raw'/'COHR-1d.json' if s=='COHR' else R.parent/'2026-09-25-现价建仓筛选'/'raw'/f'{s}.json'
        rawdaily=json.loads(dp.read_text())['chart']['result'][0];di=pd.to_datetime(rawdaily['timestamp'],unit='s',utc=True).tz_convert(NY);prev=float(pd.Series(rawdaily['indicators']['quote'][0]['close'],index=di).loc[di.date<TODAY.date()].dropna().iloc[-1]);p=meta['regularMarketPrice']
        vwap=float((((done.high+done.low+done.close)/3)*done.volume).sum()/done.volume.sum());cut=done.index[-1].time();past=[]
        for dt,g in previous.groupby(previous.index.date):past.append(dict(date=str(dt),volume=float(g[g.index.time<=cut].volume.sum())))
        volume=float(done.volume.sum());mean=np.mean([x['volume'] for x in past])
        row=dict(symbol=s,price=p,quote_at=pd.Timestamp(meta['regularMarketTime'],unit='s',tz='UTC').tz_convert(NY).isoformat(),change_pct=100*(p/prev-1),previous_close=prev,open=float(today.open.iloc[0]),day_high=meta.get('regularMarketDayHigh'),day_low=meta.get('regularMarketDayLow'),vwap_5m_proxy=vwap,completed_bar_end=(done.index[-1]+pd.Timedelta(minutes=5)).isoformat(),volume_to_cutoff=volume,prior4_same_time_volume_ratio=volume/mean,previous_volume_samples=past,last_six_completed=done.tail(6).reset_index().assign(index=lambda a:a['index'].astype(str)).to_dict('records'),bid_ask=None)
        snap.append(row)
    save('market-snapshot.json',snap)
    signal=dict(date=str(d.index[-1].date()),close=d.close.iloc[-1],high=d.high.iloc[-1],low=d.low.iloc[-1],rsi2=f.rsi2.iloc[-1],sma200=f.sma200.iloc[-1],sma50=d.close.tail(50).mean(),ema20=f.ema20.iloc[-1],atr14=f.atr.iloc[-1],entry_ceiling=d.close.iloc[-1]+.5*f.atr.iloc[-1],recent_10_daily=d.tail(10).reset_index().assign(index=lambda a:a['index'].astype(str)).to_dict('records'))
    save('signal.json',signal)
    print(json.dumps(old.clean(stats),indent=2));print('SNAPSHOT');print(pd.DataFrame(snap).drop(columns=['last_six_completed','previous_volume_samples']).to_string(index=False));print('COHR_LAST_BARS');print(pd.DataFrame(snap[0]['last_six_completed']).to_string(index=False));print('SIGNAL',signal)
if __name__=='__main__':main()
