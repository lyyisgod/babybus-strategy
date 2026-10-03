"""FOMC event description, volatility-scaled price intervals, and rate-futures arithmetic.
No calibrated directional probability. Uses already fetched daily histories.
"""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'2026-09-14-下一只SNDK/raw'
DATES='2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-07-26'.split()
def hist(s):
    x=json.loads((SOURCE/f'{s}.json').read_text())['chart']['result'][0]
    d=pd.DataFrame(x['indicators']['quote'][0],index=pd.to_datetime(x['timestamp'],unit='s',utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d'))
    d['adj']=x['indicators']['adjclose'][0]['adjclose'];d=d.loc[d.index<='2026-09-14'].dropna(subset=['adj','close'])
    for k in ['open','high','low']:d['adj_'+k]=d[k]*d.adj/d.close
    return d
def event_study(d,s):
    out=[]
    for date in DATES:
        i=d.index.get_loc(date);a=d.adj
        out.append(dict(symbol=s,decision=date,monday=str(d.index[i-2]),tuesday_ret_pct=100*(a.iloc[i-1]/a.iloc[i-2]-1),tuesday_low_below_monday_low=bool(d.adj_low.iloc[i-1]<d.adj_low.iloc[i-2]),decision_day_ret_pct=100*(a.iloc[i]/a.iloc[i-1]-1),post_two_sessions_pct=100*(a.iloc[i+2]/a.iloc[i]-1),monday_to_friday_pct=100*(a.iloc[i+2]/a.iloc[i-2]-1)))
    return out
def bands(d,s):
    a=d.adj.to_numpy();lr=np.diff(np.log(a),prepend=np.nan);rv=pd.Series(lr).rolling(20).std(ddof=1).to_numpy();h=4
    # Fixed four-session origins. Training includes only labels fully realized at the forecast origin.
    origins=np.arange(20,len(a)-h,h);z=np.array([math.log(a[i+h]/a[i])/(rv[i]*math.sqrt(h)) for i in origins])
    tests=[]
    for j,i in enumerate(origins):
        tr=(origins+h<=i)&np.isfinite(z)
        if tr.sum()<126:continue
        q80,q95=np.quantile(abs(z[tr]),[.8,.95]);actual=z[j]
        tests.append(dict(symbol=s,origin=str(d.index[i]),end=str(d.index[i+h]),n_train=int(tr.sum()),inside80=bool(abs(actual)<=q80),inside95=bool(abs(actual)<=q95)))
    q80,q95=np.quantile(abs(z[np.isfinite(z)]),[.8,.95]);price=float(d.close.iloc[-1]);sigma=float(rv[-1]*math.sqrt(h));out=dict(symbol=s,origin='2026-09-14',horizon_sessions=4,end='2026-09-18',price=price,rv20_pct=float(rv[-1]*math.sqrt(252)*100),directional_drift=0,nominal80_price=[price*math.exp(-q80*sigma),price*math.exp(q80*sigma)],nominal95_price=[price*math.exp(-q95*sigma),price*math.exp(q95*sigma)],n_historical_train=int(np.isfinite(z).sum()),oos_n=len(tests),oos_start=tests[0]['origin'],oos_end=tests[-1]['end'],oos80_coverage_pct=100*np.mean([t['inside80'] for t in tests]),oos95_coverage_pct=100*np.mean([t['inside95'] for t in tests]))
    return out,tests
def main():
    all_events=[];pred=[];tests=[];data={s:hist(s) for s in ['SPY','QQQ','SMH']}
    for s,d in data.items():
        all_events.extend(event_study(d,s));b,t=bands(d,s);pred.append(b);tests.extend(t)
    events=pd.DataFrame(all_events);events.to_csv(ROOT/'fomc_2022_2023.csv',index=False);pd.DataFrame(tests).to_csv(ROOT/'interval_oos.csv',index=False)
    es=[]
    for s,g in events.groupby('symbol'):
        es.append(dict(symbol=s,n=len(g),tuesday_down=int((g.tuesday_ret_pct<0).sum()),tuesday_under_monday_low=int(g.tuesday_low_below_monday_low.sum()),decision_day_up=int((g.decision_day_ret_pct>0).sum()),friday_above_monday=int((g.monday_to_friday_pct>0).sum()),positive_decision_followed_by_negative_two_days=int(((g.decision_day_ret_pct>0)&(g.post_two_sessions_pct<0)).sum())))
    d=data['SMH'];hits=[];last=-10
    # Fixed adverse-day analogue, selected without checking future results; exclude overlapping four-day outcomes.
    ret=d.adj.pct_change();below=d.close<d.close.rolling(50).mean()
    for i in range(50,len(d)-4):
        if i<=last+4 or not(ret.iloc[i]<=-.03 and below.iloc[i]):continue
        future=d.iloc[i+1:i+5];hits.append(dict(date=str(d.index[i]),next_day_low_below_signal_low=bool(future.adj_low.iloc[0]<d.adj_low.iloc[i]),next4_low_below_signal_low=bool(future.adj_low.min()<d.adj_low.iloc[i]),next4_return_pct=100*(future.adj.iloc[-1]/d.adj.iloc[i]-1)));last=i
    pd.DataFrame(hits).to_csv(ROOT/'semiconductor_selloff_analogues.csv',index=False)
    q=json.loads((ROOT/'latest.json').read_text())['quotes']['ZQU26.CBT'];ff=json.loads((ROOT/'raw/EFFR-NYFED.json').read_text())['refRates'][0];fprice=q['regular_price'];effr=ff['percentRate'];monthly=100-fprice;post=(30*monthly-16*effr)/14;prob=(post-effr)/.25
    assert 0<=prob<=1,'Binary hike model inconsistent with futures price'
    rate=dict(contract='ZQU26.CBT',quote=fprice,quote_at=q['regular_at'],effr=effr,effr_observation_date=ff['effectiveDate'],current_target=[ff['targetRateFrom'],ff['targetRateTo']],implied_month_average=monthly,implied_post_meeting_effr=post,implied_change_bp=(post-effr)*100,binary_hike25_weight_pct=prob*100,assumptions='EFFR 3.63% throughout Sep1-16, any change effective Sep17; 16 pre and 14 post calendar days; only hold or +25bp; no futures risk premium. Indicative market pricing, NOT official FedWatch or real-world probability.')
    out=dict(asof='2026-09-14T17:01:00-04:00',user_position='Fully cash, explicitly confirmed this turn; supersedes AVGO position.',rates=rate,events=es,intervals=pred,selloff_analogues=dict(n=len(hits),next_day_below_signal_low=sum(h['next_day_low_below_signal_low'] for h in hits),next4_below_signal_low=sum(h['next4_low_below_signal_low'] for h in hits),next4_positive=sum(h['next4_return_pct']>0 for h in hits),median_next4_pct=float(np.median([h['next4_return_pct'] for h in hits]))),limitations=['Event samples are small and not controlled for prior expectations. Full Wednesday return includes pre-announcement hours. Tuesday metrics exclude Wednesday to avoid claiming pre-announcement knowledge from daily lows.','Intervals are symmetric zero-drift four-session closing-price risk envelopes, not intraday extremes or a calibrated FOMC forecast. Empirical marginal OOS coverage does not imply conditional coverage this week.','ETF adjusted returns include distributions; current dollar bounds use unadjusted spot. Known future distributions are not separately modeled.','Analogue frequencies are descriptive and must not be treated as exact current probabilities.'])
    serialized=json.dumps(out,indent=2,default=lambda v:v.item(),allow_nan=False)
    (ROOT/'results.json').write_text(serialized);print(serialized)
if __name__=='__main__':main()
