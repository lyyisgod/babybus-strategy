"""Frozen volatility/relative-momentum screen, not a calibrated probability."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path(__file__).resolve().parent
NY = ZoneInfo('America/New_York')
TODAY = '2026-09-30'


def load(symbol, mode):
    z = json.loads((ROOT/'raw'/f'{symbol}-{mode}.json').read_text())['data']['chart']['result'][0]
    q = z['indicators']['quote'][0]
    rows = []
    for i, ts in enumerate(z['timestamp']):
        if q['close'][i] is None:
            continue
        dt = datetime.fromtimestamp(ts, NY)
        row = {k: q[k][i] for k in ['open','high','low','close','volume']}
        row.update(date=dt.date().isoformat(), time=dt.isoformat())
        rows.append(row)
    return rows, z['meta']


def wilder_rsi(close, n):
    diff = np.diff(close)
    g, l = np.maximum(diff,0), np.maximum(-diff,0)
    gain, loss = g[:n].mean(), l[:n].mean()
    for a,b in zip(g[n:],l[n:]):
        gain=(gain*(n-1)+a)/n
        loss=(loss*(n-1)+b)/n
    if gain==0 and loss==0:
        return 50.
    return 100. if loss==0 else float(100-100/(1+gain/loss))


def describe(symbol, qqqret):
    d,meta=load(symbol,'daily'); live,lmeta=load(symbol,'live')
    d=[r for r in d if r['date']<TODAY]
    assert all(r['date']<TODAY for r in d)
    if len(d)<220:
        return {'symbol':symbol,'eligible':False,'reason':'insufficient complete daily history'}
    current=[r for r in live if r['date']==TODAY]
    if not current:
        return {'symbol':symbol,'eligible':False,'reason':'no today live quote'}
    c=np.array([r['close'] for r in d]); h=np.array([r['high'] for r in d]); l=np.array([r['low'] for r in d]);v=np.array([r['volume'] or 0 for r in d])
    tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
    atr=tr[:14].mean()
    for a in tr[14:]:atr=(13*atr+a)/14
    now=current[-1]['close']; pre=now/c[-1]-1;dollar=float(np.mean(c[-21:-1]*v[-21:-1]));volratio=float(v[-1]/v[-21:-1].mean())
    eligible=bool(c[-1]>=5 and dollar>=50e6)
    # The ranking rule was frozen before any current candidates were viewed.
    score=float(100*atr/c[-1]+.5*np.clip(100*(pre-qqqret),-5,5))
    tail=np.array([r['high']/r['open']-1 for r in d[-252:] if r['open']>0])
    rr2=wilder_rsi(c,2)
    return {'symbol':symbol,'eligible':eligible,'reason':None if eligible else 'liquidity/price filter failed',
            'observed_at':current[-1]['time'],'price':float(now),'previous_close':float(c[-1]),'premarket_return_pct':float(pre*100),
            'relative_to_QQQ_pp':float(100*(pre-qqqret)), 'atr14':float(atr),'atr14_pct':float(100*atr/c[-1]),
            'score_descriptive_not_probability':score,'avg20_dollar_volume':dollar,'previous_volume_ratio':volratio,
            'sma20':float(c[-20:].mean()),'sma50':float(c[-50:].mean()),'sma200':float(c[-200:].mean()),
            'RSI2':rr2,'RSI14':wilder_rsi(c,14),'return3_pct':float(100*(c[-1]/c[-4]-1)),
            'return20_pct':float(100*(c[-1]/c[-21]-1)),
            'LRCX_branch_core':bool(c[-1]>c[-200:].mean() and rr2<=20 and c[-1]<c[-4]),
            'historical_open_to_high_5p2_touch_fraction_not_trade_winrate':float(np.mean(tail>=.052)),
            'historical_open_to_high_p90':float(np.quantile(tail,.9)),
            'today_high':max(r['high'] or r['close'] for r in current),
            'today_low':min(r['low'] or r['close'] for r in current),
            'actual_bid_ask':None,'premarket_volume_reliable':False}


if __name__=='__main__':
    syms=json.loads((ROOT/'protocol.json').read_text())['universe']
    q,_=load('QQQ','live');d,_=load('QQQ','daily');prior=[r for r in d if r['date']<TODAY][-1]['close'];qqqret=q[-1]['close']/prior-1
    rows=[]
    for s in syms:
        try:rows.append(describe(s,qqqret))
        except Exception as e:rows.append({'symbol':s,'eligible':False,'reason':str(e)})
    ranked=sorted([r for r in rows if r['eligible']],key=lambda r:r['score_descriptive_not_probability'],reverse=True)
    (ROOT/'screen-results.json').write_text(json.dumps({'as_of':datetime.now(NY).isoformat(),'universe_n':len(syms),'eligible_n':len(ranked),'candidates':ranked,'excluded':[r for r in rows if not r['eligible']]},ensure_ascii=False,indent=2)+'\n')
    (ROOT/'selected-history-symbols.json').write_text(json.dumps([r['symbol'] for r in ranked[:12]]))
    print(json.dumps([{k:r[k] for k in ['symbol','price','observed_at','premarket_return_pct','atr14_pct','score_descriptive_not_probability','historical_open_to_high_5p2_touch_fraction_not_trade_winrate','LRCX_branch_core']} for r in ranked[:20]],ensure_ascii=False,indent=2))
