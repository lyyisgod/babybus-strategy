"""Read-only market research; frozen old model inference, no orders or retraining."""
from pathlib import Path
import json, importlib.util, hashlib, ssl, urllib.request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import certifi

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'2026-09-21-双模型短线筛选'
spec=importlib.util.spec_from_file_location('dual',BASE/'dual_model.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.m.ROOT=ROOT;m.m.TODAY=pd.Timestamp('2026-09-25',tz=m.m.NY)
STOCKS=list(dict.fromkeys(m.STOCKS+'MSFT META GOOGL AMZN AAPL ORCL CRM WDAY NOW AKAM COST JPM LLY'.split()))
PROXIES=['QQQ','SPY','SMH','JNK','TLT']
def save(name,data):
    (ROOT/name).write_text(json.dumps(m.m.clean(data),ensure_ascii=False,indent=2,allow_nan=False))
def fetch(s):
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=10y&interval=1d&events=div%2Csplits'
    try:
        raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research/1.0'}),context=ssl.create_default_context(cafile=certifi.where()),timeout=25).read()
        assert json.loads(raw)['chart']['result'][0]['timestamp']
        (ROOT/'raw'/f'{s}.json').write_bytes(raw)
        return dict(symbol=s,url=url,retrieved_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest())
    except Exception as e:return dict(symbol=s,url=url,error=str(e))
def main():
    (ROOT/'raw').mkdir(parents=True,exist_ok=True)
    protocol=dict(frozen_at=datetime.now(timezone.utc).isoformat(),stocks=STOCKS,proxies=PROXIES,universe='49 saved attention stocks + 13 explicitly fixed quality/event comparators; not full market or PIT',daily_cutoff='2026-09-24',requested_horizon='actual executable purchase to tenth following session close; no calibrated probability available',model_a='saved 21-factor coefficients, no retraining; original next-open 2/3-session outcomes; only conditional relative ranking, not intraday return probabilities',model_b='close>SMA200, Wilder RSI2<=20, return3<0; next-session close auction <= signal close + .5ATR; stop/target +-2ATR; max20 subsequent sessions',cost=.002,model_hash=hashlib.sha256((BASE/'model-coefficients.json').read_bytes()).hexdigest(),absolute_gate='FAIL: no untouched PIT-universe audit or executable book; no high-confidence certification',historical_results_already_seen=True)
    save('protocol.json',protocol)
    with ThreadPoolExecutor(max_workers=6) as ex:manifest=list(ex.map(fetch,STOCKS+PROXIES))
    save('manifest.json',manifest)
    ds={};metas={};ratios={};errors={}
    for s in STOCKS+PROXIES:
        try:ds[s],metas[s],ratios[s]=m.m.load(s)
        except Exception as e:errors[s]=str(e)
    save('errors.json',errors)
    co=json.loads((BASE/'model-coefficients.json').read_text());rows=[];checks=[]
    now=datetime.now(timezone.utc)
    for s in STOCKS+PROXIES:
        if s not in ds:continue
        d=ds[s];f=m.m.factors(d);mta=metas[s];p=float(mta['regularMarketPrice']);tm=datetime.fromtimestamp(mta['regularMarketTime'],timezone.utc)
        ratio=ratios[s].iloc[-1];atr=f.atr.iloc[-1]/ratio;last=d.raw_close.iloc[-1]
        row=dict(symbol=s,signal_date=str(d.index[-1].date()),price=p,quote_at=tm.astimezone(m.m.NY).isoformat(),quote_age_seconds=(now-tm).total_seconds(),previous_close=last,change_pct=100*(p/last-1),rsi2=f.rsi2.iloc[-1],atr=atr,sma200=f.sma200.iloc[-1]/ratio,sma50=d.close.tail(50).mean()/ratio,ema20=f.ema20.iloc[-1]/ratio,ret3_pct=100*d.close.pct_change(3).iloc[-1],ret20_pct=100*d.close.pct_change(20).iloc[-1],rs20_qqq_pp=100*(d.close.pct_change(20).iloc[-1]-ds['QQQ'].close.pct_change(20).iloc[-1]),vol_ratio=d.volume.iloc[-1]/d.volume.iloc[-21:-1].mean(),avg_dollar_volume20=(d.raw_close*d.volume).tail(20).mean(),pullback_signal=bool(f.pullback.iloc[-1]),entry_ceiling=last+.5*atr,price_within_ceiling=p<=last+.5*atr,day_high=mta.get('regularMarketDayHigh'),day_low=mta.get('regularMarketDayLow'),return_to_20day_high_pct=100*(d.raw_high.tail(20).max()/p-1),stop_2atr=p-2*atr,target_2atr=p+2*atr)
        if s in STOCKS:
            x=m.factors(s,d,ds)[co['features']]
            z=np.r_[1,np.clip((x.iloc[-1].to_numpy()-co['train_mean'])/co['train_sd'],-8,8)]
            row['old_tail_ordering_value']=float(1/(1+np.exp(-np.clip(z@np.array(co['logistic'][0]),-30,30))))
            row['old_three_session_return']=float(z@np.array(co['linear']))
            cut=d.index[-15];prefix={k:v.loc[v.index<=cut] for k,v in ds.items()}
            y=m.factors(s,prefix[s],prefix)[co['features']].iloc[-1]
            assert np.allclose(x.loc[cut],y,equal_nan=True),s
            checks.append(s)
        assert d.index.max()<m.m.TODAY
        rows.append(row)
    out=pd.DataFrame(rows);out.to_csv(ROOT/'all-candidates.csv',index=False)
    stock=out[out.symbol.isin(STOCKS)].copy();stock['liquid']=(stock.previous_close>=5)&(stock.avg_dollar_volume20>=5e7)
    stock.sort_values('old_tail_ordering_value',ascending=False).to_csv(ROOT/'tail-ranking.csv',index=False)
    pull=stock[stock.pullback_signal];pull.to_csv(ROOT/'pullback-signals.csv',index=False)
    save('validation.json',dict(causal_prefix_checks=checks,all_daily_before_today=True,downloaded=len(ds),requested=len(STOCKS+PROXIES),old_coefficients_unchanged=True,training='original 2016-2022 frozen; no new fitting',audit='previously examined and survivor-biased; no new probability validation'))
    print('PULLBACK\n'+pull.to_string(index=False));print('TAIL\n'+stock.sort_values('old_tail_ordering_value',ascending=False).head(12).to_string(index=False));print('COMPARATORS\n'+out[out.symbol.isin('NVDA AVGO COHR MSFT META GOOGL AMZN AAPL AKAM LRCX MU VICR CRM WDAY QQQ SPY SMH JNK'.split())].to_string(index=False))
if __name__=='__main__':main()
