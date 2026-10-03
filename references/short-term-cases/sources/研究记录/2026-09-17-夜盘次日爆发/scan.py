"""Daily-state analog study; explicitly NOT a backtest of overnight execution.

Current universe is discretionary and contains survivors. No learned probability
or trade return is claimed. Save raw inputs; --offline reproduces this snapshot.
"""
import argparse
import json
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import certifi
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
NY = ZoneInfo('America/New_York')
ASOF = '2026-09-16'
SYMBOLS = 'CRDO ALAB COHR LITE AAOI MRVL MU SNDK NVDA AMD AVGO SMTC VICR GNRC VRT BE CEG VST ETN POWL CRWV NBIS IREN CIFR WULF RKLB ASTS PLTR OKLO'.split()
CONTEXT = ['QQQ', 'SMH', 'SPY', 'TLT', '^VIX']
FEATURES = ['r1', 'r5', 'r20', 'atr_pct', 'rvol', 'clv', 'rel5', 'qqq_r1']

def fetch(symbol, interval, offline):
    path = P / f'{symbol}-{interval}.json'
    if offline:
        return json.loads(path.read_text())['data']['chart']['result'][0]
    query = 'range=5y&interval=1d' if interval == '1d' else 'range=5d&interval=5m&includePrePost=true'
    url = f'https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?{query}&events=div%2Csplits'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25, context=ssl.create_default_context(cafile=certifi.where())) as response:
        raw = json.load(response)
    path.write_text(json.dumps({'retrieved_at': datetime.now(NY).isoformat(), 'source': url, 'data': raw}))
    return raw['chart']['result'][0]

def frame(raw):
    index = pd.to_datetime(raw['timestamp'], unit='s', utc=True).tz_convert(NY)
    f = pd.DataFrame(raw['indicators']['quote'][0], index=index).dropna(subset=['open', 'high', 'low', 'close'])
    return f

def daily(raw):
    f = frame(raw)
    f.index = f.index.strftime('%Y-%m-%d')
    f = f.loc[f.index <= ASOF].copy()
    # Yahoo OHLC is split adjusted. Apply adjclose/close to all OHLC so dividend
    # total-return factors and forward outcomes use the same adjustment basis.
    adj = raw['indicators'].get('adjclose')
    if adj:
        mapping = dict(zip(pd.to_datetime(raw['timestamp'], unit='s', utc=True).tz_convert(NY).strftime('%Y-%m-%d'), adj[0]['adjclose']))
        ratio = pd.Series(mapping).reindex(f.index) / f.close
        f[['open', 'high', 'low', 'close']] = f[['open', 'high', 'low', 'close']].mul(ratio, axis=0)
    valid = (f.low <= f[['open', 'close']].min(axis=1) + .001) & (f.high + .001 >= f[['open', 'close']].max(axis=1)) & (f.low > 0)
    if not valid.all():
        raise ValueError(f'invalid OHLC rows: {list(f.index[~valid])[:5]}')
    if f.index[-1] != ASOF:
        raise ValueError(f'stale daily date {f.index[-1]}')
    return f

def factors(f, q):
    a = f.copy()
    for n in [1, 5, 20]:
        a[f'r{n}'] = a.close.pct_change(n) * 100
    tr = pd.concat([a.high-a.low, (a.high-a.close.shift()).abs(), (a.low-a.close.shift()).abs()], axis=1).max(axis=1)
    a['atr_pct'] = tr.rolling(14).mean() / a.close * 100
    a['rvol'] = a.volume / a.volume.shift().rolling(20).mean()
    a['clv'] = (a.close-a.low) / (a.high-a.low).replace(0, np.nan)
    a['rel5'] = a.r5 - q.close.pct_change(5).reindex(a.index) * 100
    a['qqq_r1'] = q.close.pct_change().reindex(a.index) * 100
    a['next_high'] = a.high.shift(-1)/a.close - 1
    a['next_low'] = a.low.shift(-1)/a.close - 1
    a['next_close'] = a.close.shift(-1)/a.close - 1
    return a

def neighbors(a, i, k=60):
    # index i-1 outcome ends at i close and is available at decision time i.
    prior = a.iloc[:i].dropna(subset=FEATURES+['next_high', 'next_low', 'next_close'])
    x = prior[FEATURES].to_numpy()
    center = np.median(x, axis=0)
    scale = np.quantile(x, .75, axis=0)-np.quantile(x, .25, axis=0)
    scale = np.where(scale > 1e-8, scale, 1)
    distance = np.mean(((x-center)/scale-(a.iloc[i][FEATURES].to_numpy(dtype=float)-center)/scale)**2, axis=1)
    return prior.iloc[np.argsort(distance)[:min(k, len(prior))]]

def oos(a):
    preds, actual, bases = [], [], []
    for i in range(max(260, len(a)-505), len(a)-1, 5):
        if a.iloc[i][FEATURES+['next_high']].isna().any():
            continue
        ns = neighbors(a, i)
        if len(ns) < 60:
            continue
        preds.append(float((ns.next_high >= .05).mean()))
        actual.append(float(a.iloc[i].next_high >= .05))
        bases.append(float((a.iloc[:i].dropna(subset=['next_high']).next_high >= .05).mean()))
    if not preds:
        return {'n': 0, 'brier_skill': None}
    mse = np.mean((np.array(preds)-actual)**2)
    base_mse = np.mean((np.array(bases)-actual)**2)
    return {'n': len(preds), 'events': int(sum(actual)), 'brier': float(mse), 'base_brier': float(base_mse), 'brier_skill': float(1-mse/base_mse) if base_mse else None,
            'definition': 'chronological 5-session stride; next high/previous close >= 1.05; expanding prior base; no overnight-price history'}

def extended(raw):
    f = frame(raw)
    f = f.loc[f.index.strftime('%Y-%m-%d') == ASOF]
    post = f.loc[f.index.hour >= 16]
    if post.empty:
        return {}
    regular = f.loc[(f.index.hour*60+f.index.minute >= 570) & (f.index.hour < 16)]
    return {'post_price': float(post.close.iloc[-1]), 'post_time': post.index[-1].isoformat(), 'post_high': float(post.high.max()), 'post_low': float(post.low.min()),
            'regular_last_hour_pct': float((regular.close.iloc[-1]/regular.loc[regular.index.hour >= 15].open.iloc[0]-1)*100) if len(regular) else None,
            'note': '5-minute last trade proxy; not a live executable overnight bid/ask; reported zero extended volume is not interpreted as zero trading'}

def model(symbol, d, intraday):
    f = d[symbol]
    a = factors(f, d['QQQ'])
    row = a.iloc[-1]
    ns = neighbors(a, len(a)-1)
    snap = extended(intraday) if intraday else {}
    premium = snap.get('post_price', float(f.close.iloc[-1]))/f.close.iloc[-1]-1
    adjusted_high = (1+ns.next_high)/(1+premium)-1
    adjusted_low = (1+ns.next_low)/(1+premium)-1
    adjusted_close = (1+ns.next_close)/(1+premium)-1
    out = {'symbol': symbol, 'date': ASOF, 'close': float(row.close), 'high': float(row.high), 'low': float(row.low), 'open': float(row.open),
           'factors': {k: float(row[k]) for k in FEATURES}, 'factor_coverage': 1.0, **snap,
           'post_premium_pct': float(premium*100), 'ma20': float(f.close.tail(20).mean()), 'high20_prior': float(f.high.iloc[-21:-1].max()),
           'avg_dollar_volume20': float((f.close*f.volume).iloc[-21:-1].mean()), 'history_n': len(f),
           'analog_n': len(ns), 'analog_dates': list(ns.index),
           'proxy_touch5_count': int((adjusted_high >= .05).sum()), 'proxy_touch8_count': int((adjusted_high >= .08).sum()),
           'proxy_low_minus3_count': int((adjusted_low <= -.03).sum()), 'proxy_close_up_count': int((adjusted_close > 0).sum()),
           'proxy_high_p50_pct': float(adjusted_high.median()*100), 'proxy_high_p75_pct': float(adjusted_high.quantile(.75)*100),
           'proxy_low_p10_pct': float(adjusted_low.quantile(.1)*100), 'oos': oos(a)}
    out['daily_touch5_count'] = int((ns.next_high >= .05).sum())
    out['daily_touch8_count'] = int((ns.next_high >= .08).sum())
    out['applicability'] = 'daily analog only; no historical overnight conditioning'
    # A large post-close event changes the state. Ordinary daily neighbors cannot
    # price that information; refuse the tempting but misleading 0/60 output.
    if abs(premium*100) >= row.atr_pct:
        out['applicability'] = 'ABSTAIN: post-close move >= one ATR; event regime outside daily-state model'
        for key in list(out):
            if key.startswith('proxy_'):
                out[key] = None
    if symbol == 'WULF':
        out['history_warning'] = 'business regime and predecessor history differ materially; high volatility is not verified alpha'
    if snap and snap['post_low'] < row.close * .97:
        out['extended_quality_warning'] = 'large post-close excursion: possible isolated prints; high/low not accepted as support/stop without corroboration'
    if symbol == 'CRDO':
        prev = a.iloc[-2]
        out['prior_day_factors'] = {k: float(prev[k]) for k in FEATURES}
        out['actual_0916_high_from_0915_close_pct'] = float((row.high/prev.close-1)*100)
    return out

def check():
    assert abs((1.08/1.03-1)*100-4.854368932038835) < 1e-10
    n=100
    a=pd.DataFrame({k:np.arange(n,dtype=float) for k in FEATURES})
    for k in ['next_high','next_low','next_close']:a[k]=np.arange(n)/100
    first=neighbors(a,80).index.tolist()
    a.loc[81:,FEATURES]=999999
    assert first == neighbors(a,80).index.tolist() and max(first)<80
    f=pd.DataFrame({'open':[100.,101.,104.], 'high':[102.,106.,110.], 'low':[99.,100.,103.], 'close':[101.,104.,108.], 'volume':[10.,20.,30.]})
    z=factors(f,f)
    assert abs(z.next_high.iloc[0]-(106/101-1))<1e-12 and pd.isna(z.next_high.iloc[-1])
    print('PASS: entry premium adjustment; future-feature isolation; analog dates precede decision; next-session label; missing future outcome remains NaN')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:check();return
    (P/'universe.json').write_text(json.dumps({'created_at':datetime.now(NY).isoformat(),'symbols':SYMBOLS,'context':CONTEXT,'selection':'discretionary liquid AI/optics/power/high-beta watchlist + current event GNRC/VICR, not all-US universe; survivorship bias'},indent=2))
    data={};intra={};errors={}
    def run(s):
        try:return s,daily(fetch(s,'1d',args.offline)),fetch(s,'5m',args.offline),None
        except Exception as exc:return s,None,None,str(exc)
    with ThreadPoolExecutor(max_workers=5) as pool:
        for s,d,i,e in pool.map(run,SYMBOLS+CONTEXT):
            if e:errors[s]=e
            else:data[s]=d;intra[s]=i
    if 'QQQ' not in data:raise RuntimeError(errors)
    rows=[]
    for s in SYMBOLS:
        if s in data:
            try:rows.append(model(s,data,intra[s]))
            except Exception as exc:errors[s]=str(exc)
    out={'as_of':datetime.now(NY).isoformat(),'target_session':'2026-09-17','method':'8-factor robust-IQR nearest 60 same-stock daily analogs, uniform weights; current post-close premium transforms historical high/low/close outcomes. This is a counterfactual sensitivity, NOT overnight calibrated probabilities or executable backtest. OOS validates only unadjusted daily +5% high touch. No stop/target order inferable from daily bars.', 'rows':rows,'errors':errors,
         'context':{s:{'close':float(data[s].close.iloc[-1]),'day_pct':float(data[s].close.pct_change().iloc[-1]*100),**extended(intra[s])} for s in CONTEXT if s in data}}
    (P/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    flat=[]
    for r in rows:
        flat.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))}|r['factors']|{'oos_n':r['oos']['n'],'oos_brier_skill':r['oos']['brier_skill']})
    pd.DataFrame(flat).to_csv(P/'factors.csv',index=False)
    print(pd.DataFrame(flat)[['symbol','close','r1','r5','atr_pct','rvol','clv','post_price','post_premium_pct','proxy_touch5_count','proxy_touch8_count','proxy_low_minus3_count','oos_brier_skill']].round(3).to_string(index=False))
    print('ERRORS',errors)

if __name__=='__main__':main()
