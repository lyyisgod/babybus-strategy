"""Fresh market snapshots; retain frozen completed-day factors, no trading."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
BASE = ROOT / '研究记录/2026-10-02-Babybus次日筛选-1438更新'
sys.path.insert(0, str(ROOT))
from strategies.bb_washout_bounce.data import save, calendar, load_config

spec = importlib.util.spec_from_file_location('prior_refresh', BASE / 'refresh.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
proto = json.loads((BASE / 'protocol.json').read_text())
short = json.loads((BASE / 'shortlist-protocol.json').read_text())['names']
save(P / 'protocol.json', dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    stocks=proto['stocks'], proxies=proto['proxies'], completed_daily_session='2026-10-01',
    daily_factors_source=str(BASE / 'all-factors.json'),
    scope='154 fixed attention names; not whole market or point-in-time historical universe',
    target='Babybus weekly rotation entry/add; three-session washout branch separate',
    rule_changes=False, current_probability=None, orders_placed=False))
if not (P / 'live-raw').exists():
    mod.batch(proto['stocks'] + proto['proxies'], P / 'live-raw', '5d')
man = json.loads((P / 'live-raw/manifest.json').read_text())
mf = {r['symbol']: r for r in man}
if not any('file' in r for r in man):
    raise RuntimeError('No fresh quotes; do not substitute old snapshots.')
base = json.loads((BASE / 'all-factors.json').read_text())
factors = {r['ticker']: r for r in base}
prior_proxies = json.loads((BASE / 'live-proxies.json').read_text())

def frame(raw):
    return pd.DataFrame(raw['indicators']['quote'][0], index=pd.to_datetime(
        raw['timestamp'], unit='s', utc=True).tz_convert('America/New_York')).dropna(subset=['close'])

def live(s):
    m = mf[s]
    if 'error' in m:
        raise ValueError(m['error'])
    raw = json.loads((P / 'live-raw' / m['file']).read_text())['chart']['result'][0]
    at = pd.Timestamp(m['fetched_at'])
    d = frame(raw)
    d = d[(d.index + pd.Timedelta(minutes=5) <= at)]
    d = d[(d.index.hour*60+d.index.minute >= 570) & (d.index.hour*60+d.index.minute < 960)]
    c = d[d.index.strftime('%Y-%m-%d') == '2026-10-02']
    assert len(c), 'no completed current bars'
    mt = raw['meta']
    quote = pd.Timestamp(mt['regularMarketTime'], unit='s', tz='UTC')
    assert quote <= at
    prior = factors[s]['close'] if s in factors else prior_proxies[s]['prior_close']
    v = c.volume.fillna(0)
    vw = float((((c.high+c.low+c.close)/3)*v).sum()/v.sum()) if v.sum()>0 else None
    p = float(mt['regularMarketPrice'])
    rv = None
    dates = []
    if s in short:
        old = json.loads((BASE / 'shortlist-raw' / (s+'.json')).read_text())['chart']['result'][0]
        h = frame(old)
        minute = h.index.hour*60+h.index.minute
        h = h[(minute >= 570) & (minute < 960)]
        cutoff = (c.index[-1]+pd.Timedelta(minutes=5)).hour*60+(c.index[-1]+pd.Timedelta(minutes=5)).minute
        vols = []
        for day,g in h.groupby(h.index.date):
            z = g[g.index.hour*60+g.index.minute < cutoff]
            if str(day) < '2026-10-02' and len(g) == 78 and (z.index.hour*60+z.index.minute).tolist() == list(range(570,cutoff,5)):
                vols.append(float(z.volume.sum())); dates.append(str(day))
        if len(vols) >= 20:
            rv = float(v.sum()/np.mean(vols[-20:])); dates = dates[-20:]
    return dict(price=p, prior_close=prior, change_pct=100*(p/prior-1),
        quote_at=quote.tz_convert('America/New_York'), fetched_at=at,
        vwap_completed=vw, above_vwap=p>vw if vw is not None else None, completed_bar_end=c.index[-1]+pd.Timedelta(minutes=5),
        low_completed=float(c.low.min()), high_completed=float(c.high.max()),
        open=float(c.open.iloc[0]), from_open_pct=100*(p/c.open.iloc[0]-1),
        gap_pct=100*(c.open.iloc[0]/prior-1), rvol20_same_clock=rv, reference_dates=dates,
        last_two_complete_closes=c.close.iloc[-2:].tolist(),
        bid_ask=None, executable_quote=False, source=f'https://finance.yahoo.com/quote/{s}/')

rows=[]; errors={}; proxies={}
for s in proto['stocks']:
    try:
        r=dict(factors[s]); r['live']=live(s); rows.append(r)
    except Exception as ex:
        errors[s]=str(ex)
for s in proto['proxies']:
    try: proxies[s]=live(s)
    except Exception as ex: proxies[s]=dict(error=str(ex))
save(P/'all-factors.json', rows); save(P/'errors.json',errors); save(P/'proxies.json',proxies)
eligible=[r for r in rows if r['adv20']>=50e6]
wash=sorted([r for r in eligible if .35<=r['dd_high']<=.65],
    key=lambda r:(r['gate_count'],r['live']['above_vwap'],r['vol_ratio20']),reverse=True)
save(P/'washout-ranking.json',wash)
save(P/'lrcx-branch.json',[r for r in rows if r['lrcx_branch']])
save(P/'shortlist.json',[r for r in rows if r['ticker'] in short])
hashchecks={}
for folder in [P/'live-raw',BASE/'daily-raw',BASE/'shortlist-raw']:
    manifest=json.loads((folder/'manifest.json').read_text())
    hashchecks[str(folder)] = all(hashlib.sha256((folder/a['file']).read_bytes()).hexdigest()==a['sha256'] for a in manifest if 'file' in a)
assert all(hashchecks.values())
save(P/'verification.json',dict(as_of=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    requested=len(proto['stocks']),fresh_rows=len(rows),errors=errors,hashchecks=hashchecks,
    daily_unchanged_since_completed_session=True,partial_bars_excluded=True,
    same_clock_reference_uses_only_prior_sessions=True,statistical_validation=False,orders_placed=False))
for r in wash[:12]+[r for r in rows if r['ticker'] in ['COHR','APLD','QCOM']]:
    print(json.dumps(dict(ticker=r['ticker'],gates=r['gate_count'],dd=r['dd_high'],dd10=r['dd10'],rsi2=r['rsi2'],weekly_mid=r['weekly_mid'],weekly_lower=r['weekly_lower'],ma200=r['ma200'],live=r['live']),default=str),flush=True)
