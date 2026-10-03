"""Audit fresh inputs and report the separate LRCX signal and entry-price scenarios."""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import model as m

P = Path(__file__).resolve().parent
manifest = json.loads((P/'manifest.json').read_text())
assert len(manifest) == 236 and all('error' not in r for r in manifest)
for r in manifest:
    assert hashlib.sha256((P/'raw'/f"{r['symbol']}-{r['interval']}.json").read_bytes()).hexdigest() == r['sha256']
ds = {}
for s in m.STOCKS+m.PROXIES:
    try: ds[s], _ = m.load(s)
    except Exception: pass
rank = pd.read_csv(P/'ranking.csv')
coef = json.loads((P/'coefficients.json').read_text())
cols = coef['features']
factors, signals, terms, quotes = [], [], [], []
for s in rank.symbol:
    d = ds[s]
    f = m.features(d, ds).iloc[-1]
    factors.append(dict(symbol=s, **f.to_dict()))
    delta = d.close.diff().dropna()
    g = m.wilder(delta.clip(lower=0), 2).iloc[-1]
    l = m.wilder(-delta.clip(upper=0), 2).iloc[-1]
    rsi2 = 100*g/(g+l) if g+l > 0 else 50.
    sma = d.close.rolling(200).mean().iloc[-1]
    ret3 = d.close.pct_change(3).iloc[-1]
    signals.append(dict(symbol=s, rsi2=rsi2, ret3_pct=ret3*100,
                        above_sma200=d.close.iloc[-1]>sma,
                        signal=d.close.iloc[-1]>sma and rsi2<=20 and ret3<0))
    z = np.clip((f[cols].to_numpy(float)-coef['train_mean'])/coef['train_sd'], -8, 8)
    co = coef['models']['a_hit5']
    contribution = z*np.array(co['coefficients'][1:])*co['calibration'][1]
    terms.extend(dict(symbol=s, factor=k, standardized_value=z[i], log_odds_contribution=contribution[i]) for i,k in enumerate(cols))
    q = m.quote(s,d)
    t = pd.Timestamp(q['last_at'])
    q['session'] = 'after_hours' if t.normalize()==m.DAY and 16<=t.hour<20 else 'not_verified_after_hours'
    q['is_live_overnight_quote'] = False
    q['executable_bid_ask'] = None
    quotes.append(q)
pd.DataFrame(factors).to_csv(P/'current-32-factors.csv', index=False)
pd.DataFrame(signals).to_csv(P/'lrcx-signals.csv', index=False)
pd.DataFrame(terms).to_csv(P/'linear-factor-contributions.csv', index=False)
m.save('latest-quotes.json', quotes)
old = pd.read_csv(P.parent/'2026-09-21-隔夜至次日爆发/ranking.csv')
comparison = rank[['symbol','p_hit5','daily_pct','volume_ratio']].merge(old[['symbol','p_hit5','daily_pct','volume_ratio']],on='symbol',suffixes=('_new','_old'))
comparison.to_csv(P/'refresh-comparison.csv',index=False)
qualified = rank[rank.eligible]
selected = qualified.iloc[0]
result = dict(raw_hashes_checked=len(manifest), daily_date=str(m.DAY),
    features=len(cols), calculated=len(rank), eligible=len(qualified),
    relative_first=selected.symbol, a_first=qualified.sort_values('a_hit5',ascending=False).iloc[0].symbol,
    b_first=qualified.sort_values('b_hit5',ascending=False).iloc[0].symbol,
    lrcx_qualified=[x for x in signals if x['signal'] and x['symbol'] in set(qualified.symbol)],
    max_probability_change=float((comparison.p_hit5_new-comparison.p_hit5_old).abs().max()),
    manifest_start=min(r['fetched_at'] for r in manifest), manifest_end=max(r['fetched_at'] for r in manifest),
    overnight_executable_quote_coverage=0,
    warning='No historical overnight entry prices. No probability or confidence interval for actual overnight fills can be claimed.')
m.save('refresh-audit.json', result)
print(json.dumps(m.clean(result),ensure_ascii=False,indent=2))
print(qualified[['symbol','last','last_at','p_hit5','p_hit10','p_close3','p_down3']].head(8).to_string(index=False))
