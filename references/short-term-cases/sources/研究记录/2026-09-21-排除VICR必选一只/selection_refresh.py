"""Reuse frozen daily models; refresh entry eligibility, never manufacture event probabilities."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util, json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'2026-09-21-隔夜至次日爆发'
spec=importlib.util.spec_from_file_location('prior_model', BASE/'model.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def save(name, value):
    (ROOT/name).write_text(json.dumps(m.clean(value), ensure_ascii=False, indent=2, allow_nan=False))

def quote(symbol, close, fetched_at):
    path=ROOT/'raw'/f'{symbol}-5m.json'
    r=json.loads(path.read_text())['chart']['result'][0]
    b=pd.DataFrame(r['indicators']['quote'][0], index=pd.to_datetime(r['timestamp'],unit='s',utc=True).tz_convert(m.NY))
    now=pd.Timestamp(fetched_at).tz_convert(m.NY)
    b=b[(b.index<=now)].dropna(subset=['close'])
    t=b.index[-1];p=float(b.close.iloc[-1]);age=(now-t).total_seconds()/60
    is_post=t.normalize()==m.DAY and 16<=t.hour<20
    return dict(last=p, last_at=t, fetched_at=now, age_minutes=age,
                after_hours_change_pct=100*(p/close-1), session='after_hours' if is_post else 'regular_or_stale',
                fresh_after_hours=is_post and 0<=age<=15,
                executable_bid_ask=None, quote_source=f'https://finance.yahoo.com/quote/{symbol}/')

def main():
    original=json.loads((BASE/'ranking.json').read_text())
    manifests=json.loads((ROOT/'refresh-manifest.json').read_text())
    lookup={r['symbol']:r for r in manifests if 'error' not in r}
    assert len(lookup)==len(manifests), 'Quote fetch errors must be resolved or disclosed'
    for row in lookup.values():
        assert hashlib.sha256((ROOT/'raw'/f"{row['symbol']}-5m.json").read_bytes()).hexdigest()==row['sha256']
    coef=json.loads((BASE/'coefficients.json').read_text())
    cols=coef['features'];mu=np.asarray(coef['train_mean']);sd=np.asarray(coef['train_sd'])
    ds={};errors={}
    for s in m.STOCKS+m.PROXIES:
        try: ds[s],_=m.load(s)
        except Exception as e: errors[s]=str(e)
    rows=[];lrcx=[];contributions=[];factors=[];diffs=[]
    interactions=[('ret1','volume_ratio'),('range_position','volume_ratio'),('ret1','QQQ_r1'),('atr_pct','VIX_level'),('ret20','rs20')]
    for old in original:
        s=old['symbol'];d=ds[s];f=m.features(d,ds).iloc[-1]
        z=np.clip((f[cols].to_numpy(float)-mu)/sd,-8,8)
        a=np.r_[1,z];b=np.r_[1,z,z*z,*[z[cols.index(l)]*z[cols.index(r)] for l,r in interactions]]
        for kind,design in [('a',a),('b',b)]:
            for target in m.TARGETS:
                co=coef['models'][f'{kind}_{target}'];cal=co['calibration']
                p=m.sigmoid(cal[0]+cal[1]*(design@co['coefficients']))
                diffs.append(abs(float(p)-old[f'{kind}_{target}']))
        factors.append(dict(symbol=s,**f.to_dict()))
        ac=coef['models']['a_hit5'];terms=z*np.asarray(ac['coefficients'][1:])*ac['calibration'][1]
        contributions.extend(dict(symbol=s,factor=k,standardized_value=z[i],calibrated_log_odds_contribution=terms[i]) for i,k in enumerate(cols))
        row=old.copy();row.update(quote(s,float(d.raw_close.iloc[-1]),lookup[s]['fetched_at']))
        reasons=[]
        if not row['eligible']:reasons.append('liquidity')
        if s=='VICR':reasons.append('user_excluded')
        if row['daily_pct']>=10:reasons.append('regular_gain_ge_10pct')
        if row['after_hours_change_pct']>=5:reasons.append('after_hours_gain_ge_5pct')
        if not row['fresh_after_hours']:reasons.append('quote_stale_or_not_after_hours')
        row['overlay_eligible']=not reasons;row['exclusion_reasons']=reasons
        rows.append(row)
        delta=d.close.diff().dropna()
        gain=m.wilder(delta.clip(lower=0),2);loss=m.wilder(-delta.clip(upper=0),2)
        g,l=gain.iloc[-1],loss.iloc[-1]
        rsi=100*g/(g+l) if g+l>0 else 50.
        sma200=d.close.rolling(200).mean().iloc[-1]
        ret3=d.close.pct_change(3).iloc[-1]
        signal=d.close.iloc[-1]>sma200 and rsi<=20 and ret3<0
        lrcx.append(dict(symbol=s,close=d.raw_close.iloc[-1],rsi2=rsi,ret3_pct=ret3*100,
                         above_sma200=d.close.iloc[-1]>sma200,sma200_raw=sma200/(d.close.iloc[-1]/d.raw_close.iloc[-1]),
                         lrcx_signal=signal,liquidity_eligible=old['eligible'],atr14=old['atr']))
    assert max(diffs)<1e-10, 'Frozen-model reproduction mismatch'
    df=pd.DataFrame(rows).sort_values('p_hit5',ascending=False)
    selected=df[df.overlay_eligible].copy()
    for k in ['p_hit5','a_hit5','b_hit5','p_hit10']:
        selected[f'{k}_rank']=selected[k].rank(ascending=False,method='min').astype(int)
    assert 'VICR' not in set(selected.symbol)
    assert (selected.daily_pct<10).all() and (selected.after_hours_change_pct<5).all()
    df.to_csv(ROOT/'all-ranking.csv',index=False);selected.to_csv(ROOT/'eligible-ranking.csv',index=False)
    pd.DataFrame(lrcx).to_csv(ROOT/'lrcx-signals.csv',index=False)
    pd.DataFrame(factors).to_csv(ROOT/'latest-32-factors.csv',index=False)
    pd.DataFrame(contributions).to_csv(ROOT/'linear-model-contributions.csv',index=False)
    save('ranking.json',df.to_dict('records'))
    save('latest-quotes.json',[dict(symbol=s,**quote(s,float(ds[s].raw_close.iloc[-1]),lookup[s]['fetched_at'])) for s in ['AAOI','CRDO','CIFR','NBIS','ALAB','FSLY','COHR','SMTC','QQQ','SMH']])
    save('validation.json',dict(status='pass',quote_hashes_checked=len(manifests),max_reproduced_prediction_difference=max(diffs),
         daily_models_retrained=False,original_count=len(original),original_liquid=int(df.eligible.sum()),
         new_eligible=len(selected),top1=str(selected.iloc[0].symbol),
         a_top1=str(selected.sort_values('a_hit5',ascending=False).iloc[0].symbol),
         b_top1=str(selected.sort_values('b_hit5',ascending=False).iloc[0].symbol),
         lrcx_signals=[r for r in lrcx if r['lrcx_signal'] and r['liquidity_eligible']],data_errors=errors,
         warning='A/B are correlated models; agreement is not a new probability. Human no-chase thresholds and current entry/stop overlay have not been independently backtested. Current quote is not a fill or overnight venue feed.'))
    print(selected[['symbol','close','daily_pct','last','last_at','after_hours_change_pct','p_hit5','p_hit10','p_down3','a_hit5_rank','b_hit5_rank']].head(12).to_string(index=False))
    print(json.dumps(m.clean({'lrcx_signals':[r for r in lrcx if r['lrcx_signal'] and r['liquidity_eligible']], 'count':len(selected),'max_prediction_diff':max(diffs)}),ensure_ascii=False))

if __name__=='__main__': main()
