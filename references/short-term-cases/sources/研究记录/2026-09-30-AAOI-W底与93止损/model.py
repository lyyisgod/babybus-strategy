"""Frozen descriptive analogue study. No calibrated current win probability."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROTOCOL = json.loads((BASE / 'protocol.json').read_text())


def load(symbol):
    r = json.loads((BASE / 'raw' / f'{symbol}-daily.json').read_text())['chart']['result'][0]
    df = pd.DataFrame(r['indicators']['quote'][0])
    df.index = pd.to_datetime(r['timestamp'], unit='s', utc=True).tz_convert('America/New_York').date
    return df.dropna(subset=['open', 'high', 'low', 'close'])


def wilder(s, period):
    out = pd.Series(np.nan, index=s.index)
    first = s.first_valid_index()
    start = s.index.get_loc(first) + period - 1
    if start >= len(s):
        return out
    out.iloc[start] = s.iloc[start-period+1:start+1].mean()
    for i in range(start+1, len(s)):
        out.iloc[i] = (out.iloc[i-1]*(period-1) + s.iloc[i])/period
    return out


def indicators(df):
    c = df.close
    tr = pd.concat([df.high-df.low, (df.high-c.shift()).abs(), (df.low-c.shift()).abs()], axis=1).max(axis=1)
    f = pd.DataFrame(index=df.index)
    f['return5'] = c.pct_change(5)
    f['return20'] = c.pct_change(20)
    f['distance_ma20'] = c/c.rolling(20).mean()-1
    f['distance_ma200'] = c/c.rolling(200).mean()-1
    delta = c.diff()
    for n in [2, 14]:
        gain = wilder(delta.clip(lower=0), n)
        loss = wilder(-delta.clip(upper=0), n)
        f[f'rsi{n}'] = 100 - 100/(1+gain/loss)
    f['RSI14/100'] = f.rsi14/100
    f['atr14'] = wilder(tr, 14)
    f['ATR14/close'] = f.atr14/c
    f['distance_low20'] = c/df.low.rolling(20).min()-1
    return f


def path(df, i, horizon, cost):
    entry = float(df.open.iloc[i+1])
    stop = entry * (PROTOCOL['stop']/PROTOCOL['quote_reference'])
    target = entry * (PROTOCOL['target']/PROTOCOL['quote_reference'])
    hit_any = bool(df.high.iloc[i+1:i+horizon+1].max() >= target)
    ambiguous = False
    for j in range(i+1, i+horizon+1):
        r = df.iloc[j]
        if r.open <= stop:
            kind, exit_price = 'stop', r.open
        elif r.open >= target:
            kind, exit_price = 'target', target
        elif r.low <= stop:
            kind, exit_price = 'stop', stop
            ambiguous = bool(r.high >= target)
        elif r.high >= target:
            kind, exit_price = 'target', target
        else:
            continue
        return dict(kind=kind, net_return=float(exit_price/entry-1-cost),
                    held_sessions=j-i, ambiguous=ambiguous, hit_any=hit_any)
    return dict(kind='timeout', net_return=float(df.close.iloc[i+horizon]/entry-1-cost),
                held_sessions=horizon, ambiguous=False, hit_any=hit_any)


def wilson(k, n):
    z = 1.96
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def summarize(df, ids):
    out = {}
    for h in PROTOCOL['horizons_sessions']:
        rows = [path(df, i, h, .002) for i in ids]
        ret = np.array([r['net_return'] for r in rows])
        counts = {k: sum(r['kind'] == k for r in rows) for k in ['target','stop','timeout']}
        out[str(h)] = dict(n=len(rows), **counts,
            target_first_rate=counts['target']/len(rows),
            target_first_wilson95_descriptive=wilson(counts['target'],len(rows)),
            target_touched_ignoring_stop=sum(r['hit_any'] for r in rows)/len(rows),
            mean_net=float(ret.mean()), median_net=float(np.median(ret)),
            positive_net_rate=float((ret>0).mean()),
            mean_net_double_cost=float(ret.mean()-.002),
            ambiguous=sum(r['ambiguous'] for r in rows),
            mean_held_sessions=float(np.mean([r['held_sessions'] for r in rows])))
    return out


def main():
    raw = load('AAOI')
    complete = raw.loc[[d.isoformat() < '2026-09-30' for d in raw.index]].copy()
    f = indicators(complete)
    names = PROTOCOL['features']
    mature = f.iloc[:-20][names].dropna()
    scale = (mature.quantile(.75)-mature.quantile(.25)).replace(0,1)
    median = mature.median()
    x = (mature-median)/scale
    result = {'protocol':PROTOCOL,'complete_data_range':[str(complete.index[0]),str(complete.index[-1])],
              'mature_candidate_n':len(mature),'quote':PROTOCOL['quote_reference']}
    current = raw.copy()
    current.loc[current.index[-1], 'close'] = PROTOCOL['quote_reference']
    queries = {'last_completed':f.iloc[-1], 'partial_today_proxy':indicators(current).iloc[-1]}
    for key, q in queries.items():
        distance = np.sqrt((((x-(q[names]-median)/scale)**2).sum(axis=1)))
        ids, selected = [], []
        for day in distance.sort_values().index:
            i = complete.index.get_loc(day)
            if all(abs(i-j)>=21 for j in ids):
                ids.append(i)
                selected.append({'signal_date':str(day),'distance':float(distance.loc[day])})
            if len(ids) == PROTOCOL['analogues']['n']:
                break
        result[key] = {'features':{k:float(q[k]) for k in f.columns},
                       'selected':selected,'outcomes':summarize(complete,ids)}
    valid_ids = [complete.index.get_loc(d) for d in mature.index]
    baseline = valid_ids[::21]
    result['unconditional_nonoverlap_baseline'] = summarize(complete,baseline)
    result['peer_snapshot'] = {}
    for symbol in ['AAOI','COHR','LITE','QQQ','SMH']:
        d = load(symbol)
        prior = d.loc[[x.isoformat() < '2026-09-30' for x in d.index]].iloc[-1]
        price = float(json.loads((BASE/'raw'/f'{symbol}-daily.json').read_text())['chart']['result'][0]['meta']['regularMarketPrice'])
        result['peer_snapshot'][symbol] = {'price':price,'prior_close':float(prior.close),
                                         'change':price/prior.close-1}
    quote = PROTOCOL['quote_reference']
    result['trade_math'] = {'risk_usd':quote-93,'reward_usd':110-quote,
        'risk_pct':1-93/quote,'reward_pct':110/quote-1,'rr':(110-quote)/(quote-93),
        'breakeven_no_cost':(quote-93)/17,'breakeven_cost_02':(quote-93+quote*.002)/17,
        'max_entry_rr25':(110+2.5*93)/3.5,
        'risk_atr_ratio':(quote-93)/f.atr14.iloc[-1]}
    (BASE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for k in ['trade_math','last_completed','partial_today_proxy','unconditional_nonoverlap_baseline','peer_snapshot']:
        item=result[k]
        if k in queries:
            item={x:y for x,y in item.items() if x!='selected'}
        print(k,json.dumps(item,ensure_ascii=False))
    complete.tail(35).to_csv(BASE/'recent_daily.csv')


if __name__ == '__main__':
    main()
