"""Apply previously frozen 21-factor coefficients and the LRCX signal to Sep 22 bars."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    out=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out

daily=module('current_daily',ROOT/'model.py')
dual=module('prior_dual',BASE/'2026-09-21-双模型短线筛选/dual_model.py')
coeff=json.loads((BASE/'2026-09-21-双模型短线筛选/model-coefficients.json').read_text())
ds={}
for symbol in dict.fromkeys(daily.STOCKS+dual.STOCKS+daily.PROXIES):
    ds[symbol]=daily.load(symbol)[0]

columns=coeff['features']
mu=np.array(coeff['train_mean']);sd=np.array(coeff['train_sd'])
coef=np.array(coeff['logistic']);linear=np.array(coeff['linear'])
rows=[]
for symbol in dual.STOCKS:
    d=ds[symbol]
    x=dual.factors(symbol,d,ds).iloc[-1]
    assert x[columns].notna().all(),symbol
    z=np.r_[1,np.clip((x[columns].to_numpy(float)-mu)/sd,-8,8)]
    probs=1/(1+np.exp(-np.clip(coef@z,-30,30)))
    rows.append(dict(symbol=symbol,signal_date=str(d.index[-1].date()),p_two_session_net5=float(probs[0]),
                     p_three_session_net8=float(probs[1]),expected_three_session_net=float(linear@z),
                     rsi2=float(x.rsi2*100),return3_pct=float(x.ret3*100),
                     note='Fixed Sep 21 coefficients; next regular open through second/third regular close, not overnight entry or tomorrow intraday high'))
rows.sort(key=lambda a:a['p_two_session_net5'],reverse=True)
(ROOT/'21-factor-ranking.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))

signals=[]
for symbol in daily.STOCKS:
    d=ds[symbol];f=dual.m.factors(d)
    if not bool(f.pullback.iloc[-1]):continue
    px=float(d.raw_close.iloc[-1]);ratio=float(d.close.iloc[-1]/px)
    atr=float(f.atr.iloc[-1]/ratio)
    signals.append(dict(symbol=symbol,signal_date=str(d.index[-1].date()),close=px,
                        rsi2=float(f.rsi2.iloc[-1]),return3_pct=float(f.r3.iloc[-1]*100),
                        atr14=atr,next_close_auction_ceiling=px+.5*atr,
                        note='Original LRCX entry is next regular close auction and hold up to 20 sessions; not tomorrow premarket-to-intraday target'))
(ROOT/'lrcx-type-signals.json').write_text(json.dumps(signals,ensure_ascii=False,indent=2))
print('21-factor top',[(r['symbol'],round(r['p_two_session_net5'],4)) for r in rows[:10]])
print('LRCX-type',signals)
