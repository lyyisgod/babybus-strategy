"""Cross-sector snapshot and factor attribution. No calibrated probability claim.
Reuses the verified price-indicator routines from this session's ten-stock study.
"""
import importlib.util,json,sys,ssl,urllib.request
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import certifi

ROOT=Path(__file__).resolve().parent
OFFLINE='--offline' in sys.argv
NOW=datetime.now(timezone.utc);NY=ZoneInfo('America/New_York')
if OFFLINE:NOW=datetime.fromisoformat(json.loads((ROOT/'manifest.json').read_text())['asof'])
spec=importlib.util.spec_from_file_location('price_core',ROOT.parent/'2026-09-17-截图十股建仓/model.py')
core=importlib.util.module_from_spec(spec)
args=sys.argv;sys.argv=[sys.argv[0]]
spec.loader.exec_module(core);sys.argv=args
core.ROOT=ROOT;core.NOW=NOW;core.OFFLINE=OFFLINE
GROUPS={
'compute_memory':'NVDA AVGO AMD MU SNDK MRVL ALAB TSM WDC STX',
'connectivity_equipment':'CRDO LITE COHR AAOI AMAT LRCX KLAC ONTO CAMT TER SMTC VICR',
'ai_infra_power':'CLS FN VRT NBIS CRWV IREN BE GEV ETN CEG VST',
'other_growth':'RKLB ASTS PLTR IOVA ABCL CDNA TEAM AMPL RNG ATRC PBF'}
POOL=[s for a in GROUPS.values() for s in a.split()]
ETFS='SPY QQQ IWM RSP SMH SOXX IGV XBI XHE XLK XLF XLE XLI XLU XLY XLP XLV XLB XLRE XLC TLT IEF HYG LQD'.split()
MACRO=['^VIX','^TNX','DX-Y.NYB','BZ=F','GC=F']
SYMBOLS=POOL+ETFS+MACRO
def beta(s,b):
    j=pd.concat([s.pct_change().rename('y'),b.pct_change().rename('x')],axis=1).dropna().tail(120)
    if len(j)<80:return None,None,None
    x=np.c_[np.ones(len(j)),j.x];coef=np.linalg.lstsq(x,j.y,rcond=None)[0]
    fitted=x@coef;den=((j.y-j.y.mean())**2).sum();r2=1-((j.y-fitted)**2).sum()/den if den else None
    return float(coef[1]),float(coef[0]),float(r2)
def intraday(s,d):
    intr,m=core.history(s,'intraday');tm=datetime.fromtimestamp(m['regularMarketTime'],NY)
    minute=intr.index.hour*60+intr.index.minute
    # US stocks and ETFs; macro uses provider metadata without assuming stock hours.
    today=intr.loc[(intr.index.date==NOW.astimezone(NY).date())&(minute>=570)&(minute<960)]
    complete=today.loc[today.index+pd.Timedelta(minutes=5)<=pd.Timestamp(NOW).tz_convert(NY)]
    p=float(m['regularMarketPrice']);prev=float(d.close.iloc[-1]);vwap=None;vr=None;n=0
    if len(complete) and complete.volume.sum()>0:
        vwap=float((((complete.high+complete.low+complete.close)/3)*complete.volume).sum()/complete.volume.sum())
        endmin=complete.index[-1].hour*60+complete.index[-1].minute
        reg=intr.loc[(minute>=570)&(minute<=endmin)]
        totals=[float(g.volume.sum()) for day,g in reg.groupby(reg.index.date) if day<NOW.astimezone(NY).date()]
        n=len(totals);vr=float(complete.volume.sum()/np.mean(totals)) if n and np.mean(totals)>0 else None
    opening=float(today.open.iloc[0]) if len(today) else None
    quality=[];high=m.get('regularMarketDayHigh');low=m.get('regularMarketDayLow')
    if opening is not None and high is not None and low is not None and not low-0.0001<=opening<=high+0.0001:
        quality.append('Opening bar inconsistent with provider daily range; opening/gap excluded')
        opening=None
    if s in MACRO:
        opening=vwap=vr=None;n=0
        quality.append('Stock-session intraday metrics suppressed for macro series; futures change uses previous daily bar, not guaranteed exchange settlement')
    return dict(symbol=s,time_et=tm.isoformat(),price=p,prev_close=prev,change_pct=100*(p/prev-1),high=high,low=low,volume=m.get('regularMarketVolume'),vwap_proxy=vwap,same_time_volume_ratio=vr,volume_comparison_days=n,open=opening,quality_notes='; '.join(quality),market_cap=m.get('marketCap'),regular_market_time_fresh=(NOW-tm.astimezone(timezone.utc)).total_seconds()<900)
def main():
    (ROOT/'raw').mkdir(exist_ok=True)
    if not OFFLINE:(ROOT/'universe.json').write_text(json.dumps({'fixed_before_fetch':True,'asof':NOW.isoformat(),'groups':GROUPS,'etfs':ETFS,'macro':MACRO,'purpose':'Purposive growth universe, not all listed equities; comparison groups not independent.'},indent=2))
    jobs=[(s,k) for s in SYMBOLS for k in ['daily','intraday']]
    with ThreadPoolExecutor(max_workers=7) as ex:manifest=list(ex.map(core.fetch,jobs))
    if not OFFLINE:(ROOT/'manifest.json').write_text(json.dumps({'asof':NOW.isoformat(),'requests':manifest},indent=2))
    ds={};fs={};qs={};errors={};rows=[]
    for s in SYMBOLS:
        try:
            ds[s]=core.history(s)[0];fs[s]=core.factors(ds[s]);qs[s]=intraday(s,ds[s])
        except Exception as e:errors[s]=str(e)
    for s,d in ds.items():
        if s not in qs:continue
        z=fs[s].iloc[-1];q=qs[s]
        r=dict(symbol=s,group=next((g for g,ss in GROUPS.items() if s in ss.split()),'macro' if s in MACRO else 'etf'),daily_date=str(d.index[-1]),**z.to_dict())
        r['previous_close']=r.pop('price');r.update(q)
        r['extension_now_atr']=(q['price']-z.ema20)/z.atr14
        r['drawdown52_now_pct']=100*(q['price']/z.high252_inclusive-1)
        r['gap_pct']=100*(q['open']/q['prev_close']-1) if q['open'] else None
        r['above_vwap']=q['price']>q['vwap_proxy'] if q['vwap_proxy'] else None
        r['live_above_ema20']=q['price']>z.ema20
        r['live_above_ema50']=q['price']>z.ema50
        r['new52high']=q['price']>z.high252_inclusive
        for b in ['SPY','QQQ','SMH']:
            if b not in ds:continue
            j=pd.concat([d.adj.rename('s'),ds[b].adj.rename('b')],axis=1).dropna()
            r[f'rs20_{b}_pp']=100*((j.s.iloc[-1]/j.s.iloc[-21]-1)-(j.b.iloc[-1]/j.b.iloc[-21]-1)) if len(j)>20 else None
            r[f'rs_today_{b}_pp']=q['change_pct']-qs[b]['change_pct']
        be,alpha,r2=beta(d.adj,ds['QQQ'].adj)
        r.update(beta120_qqq=be,r2_qqq=r2,market_model_residual_today_pp=q['change_pct']-100*alpha-be*qs['QQQ']['change_pct'] if be is not None else None)
        rows.append(r)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'factors.csv',index=False)
    pd.DataFrame(qs.values()).to_csv(ROOT/'quotes.csv',index=False)
    summaries={}
    for g in GROUPS:
        x=df.loc[df.group==g]
        summaries[g]={'n':len(x),'up':int((x.change_pct>0).sum()),'above_vwap':int(x.above_vwap.eq(True).sum()),'above_ema20':int(x.live_above_ema20.eq(True).sum()),'median_today_pct':float(x.change_pct.median()),'median_ret20_pct':float(x.ret20.median()),'median_time_volume_ratio':float(x.same_time_volume_ratio.median())}
    out={'asof':NOW.isoformat(),'requested_symbols':len(SYMBOLS),'candidate_count':len(POOL),'success_count':len(qs),'errors':errors,'groups':summaries,'notes':['Snapshot not synchronized ticks; market data may be delayed.','Daily factors end previous session. Intraday volume uses four prior sessions only.','QQQ OLS residual is an association, not causal news attribution. Macro futures can have different settlement/roll conventions.','No calibrated explosion probability; no proxy substituted for earnings revisions, options skew or prime brokerage leverage.']}
    (ROOT/'results.json').write_text(json.dumps(core.clean(out),indent=2))
    print(df[['symbol','price','change_pct','ret20','ret63','same_time_volume_ratio','extension_now_atr','above_vwap','live_above_ema50','market_model_residual_today_pp']].round(3).to_string(index=False))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
