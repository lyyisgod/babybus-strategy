"""复用冻结Babybus价格闸门和原严格策略，最新完整周五收盘研究。"""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
ROOT = P.parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('prior_scan', ROOT / '研究记录/2026-10-01-夜盘Babybus全面筛选/scan.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.P = P
m.PROTO = json.loads((P / 'protocol.json').read_text())
from strategies.bb_washout_bounce.data import clean

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
market, metadata, errors, last = m.load_market(P / 'daily-raw', m.CFG, now)
assert str(last.date()) == m.PROTO['daily_cutoff'] == '2026-10-02'
cal = m.calendar(m.CFG)
assert cal.next_session(last) == pd.Timestamp(m.PROTO['target_session'])
m.save(P / 'loader-audit.json', dict(as_of=now, last_completed=last, errors=errors))
rows, factors, factor_errors = [], {}, {}
for symbol in m.PROTO['stocks']:
    try:
        d = market[symbol]
        assert d.index[-1] == last, 'latest complete session missing'
        f = m.light(d, market['SMH'], cal)
        factors[symbol] = f
        r = f.iloc[-1].to_dict()
        r.update(ticker=symbol, session=last, gates=m.gates(r),
                 high_today=float(d.high.iloc[-1]), low_today=float(d.low.iloc[-1]),
                 ma20=float(d.adjclose.tail(20).mean() / d.factor.iloc[-1]),
                 high20_prior=float(d.high.iloc[-21:-1].max()),
                 market_cap=metadata[symbol]['provider'].get('marketCap'),
                 quote_currency=metadata[symbol]['provider'].get('currency'),
                 company_name=metadata[symbol]['provider'].get('longName'))
        r['gate_count'] = sum(r['gates'].values())
        r.update(m.live(symbol))
        p = r['last_extended_price']
        r['extended_ret_vs_close'] = p / r['close'] - 1 if p else None
        r['reference_entry_cap'] = np.floor(min(p * 1.005, r['close'] * 1.02) * 100) / 100 if p else None
        r['price_only_missing'] = [k for k in ['dd_high', 'rv90', 'weekly_vol', 'dist_low20', 'weekly_mid', 'vol_ratio20'] if pd.isna(r[k])]
        rows.append(r)
    except Exception as exc:
        factor_errors[symbol] = str(exc)
m.save(P / 'all-factors.json', rows)
m.save(P / 'factor-errors.json', factor_errors)
eligible = [r for r in rows if r['adv20'] >= 50e6]
washout = sorted([r for r in eligible if .35 <= r['dd_high'] <= .65],
                 key=lambda r: (r['gate_count'], r['close'] > r['vwap5m_proxy'], r['vol_ratio20']), reverse=True)
momentum = sorted([r for r in eligible if r['momentum_analog'] and r['close'] > r['vwap5m_proxy']],
                  key=lambda r: (r['vol_ratio20'], r['relative20']), reverse=True)
lrcx = [r for r in eligible if r['lrcx_branch']]
for name, rank in [('washout-ranking', washout), ('momentum-ranking', momentum), ('lrcx-branch', lrcx)]:
    m.save(P / (name + '.json'), rank)
securities = json.loads((ROOT / 'config/bb_universe.json').read_text())['securities']
inputs = {k: m.read_records(ROOT / 'data/bb_washout_bounce' / (k + '.jsonl'))
          for k in ['fundamentals', 'macro', 'events', 'event_coverage', 'supports', 'conflicts']}
engine = m.Engine(market, securities, inputs, m.CFG)
strict = m.daily_report(engine, last, P / 'strict-original', now=now)
m.save(P / 'strict-summary.json', strict)

# 信用轮动版独立于3日洗盘版。最新完整周线已是10/2，不沿用9/25。
j = market['JNK'].adjclose
monthly = j.resample('ME').last()
monthly = monthly[monthly.index <= last]
weekly = j.resample('W-FRI').last()
weekly = weekly[weekly.index <= last]
credit = dict(as_of=now, session=last, jnk_ret20=j.iloc[-1] / j.iloc[-21] - 1,
              jnk_ret5=j.iloc[-1] / j.iloc[-6] - 1, completed_month=monthly.index[-1],
              bear_month=bool(monthly.tail(3).mean() < monthly.tail(10).mean()),
              completed_week=weekly.index[-1], completed_week_close=weekly.iloc[-1],
              prior8week_low=weekly.iloc[-9:-1].min(),
              weekly_break=bool(weekly.iloc[-1] < weekly.iloc[-9:-1].min() * .995),
              credit_state='UNKNOWN', nfp_protection_expired_after='2026-10-02 close')
macro_manifest = {r['series']: r for r in json.loads((P / 'macro-manifest.json').read_text())}
for series in ['BAMLH0A0HYM2', 'DFII10']:
    try:
        raw = pd.read_csv(P / (series + '.csv'))
        raw.index = pd.to_datetime(raw.iloc[:, 0])
        values = pd.to_numeric(raw[series], errors='coerce').dropna().loc[:last]
        date = values.index[-1]
        base = cal.session_offset(cal.date_to_session(date, direction='previous'), -20)
        prior = values.loc[:base]
        credit[series] = dict(observed_date=date, value=values.iloc[-1], base_date=prior.index[-1],
                              base_value=prior.iloc[-1], change20_bp=(values.iloc[-1] - prior.iloc[-1]) * 100,
                              source=f'https://fred.stlouisfed.org/series/{series}',
                              fetched_at=macro_manifest[series]['fetched_at'], current_download_not_historical_vintage=True)
    except Exception as exc:
        credit[series] = dict(error=str(exc))
if credit['weekly_break'] or credit['bear_month'] or credit.get('BAMLH0A0HYM2', {}).get('change20_bp', -999) >= 50:
    credit['credit_state'] = 'OFF'
for symbol in ['SPY', 'QQQ', 'SMH', 'JNK', '^VIX', '^TNX', 'BZ=F', 'NQ=F', 'BTC-USD']:
    if symbol in market:
        d = market[symbol]
        credit[symbol] = dict(session=d.index[-1], close=d.close.iloc[-1],
                             dayreturn=d.adjclose.iloc[-1] / d.adjclose.iloc[-2] - 1,
                             return20=d.adjclose.iloc[-1] / d.adjclose.iloc[-21] - 1,
                             notes='regular-equity session filter; not current Saturday crypto/futures quote' if symbol in ['BTC-USD', 'NQ=F'] else None)
m.save(P / 'credit-context.json', credit)

# 冻结的价格类比只作描述：前收买入代理，不是可交易夜盘概率。
# 严格条件事件多寡、基本面与隔夜盘口未进入此样本。
selected_symbols = {r['ticker'] for r in washout[:12]} | {'IREN', 'APLD', 'AAOI', 'COHR', 'CBRS', 'CIFR', 'CRWV', 'NBIS'}
stats, events = [], []
for symbol in sorted(selected_symbols):
    if symbol not in factors:
        continue
    for kind in ['washout_analog', 'momentum_analog']:
        f = factors[symbol]
        subset = f.loc['2024-01-01':]
        subset = subset[subset[kind] & subset.next_close_net.notna() & (subset.adv20 >= 50e6)]
        previous, local = None, []
        for day, row in subset.iterrows():
            if previous is not None and day <= cal.next_session(previous):
                continue
            previous = day
            exit_session = cal.next_session(day)
            assert exit_session <= last
            local.append(dict(ticker=symbol, kind=kind, signal_session=day, exit_session=exit_session,
                              high=row.next_high, low=row.next_low, close_net=row.next_close_net))
        events += local
        n = len(local)
        hits = sum(e['high'] >= .05 for e in local)
        stats.append(dict(ticker=symbol, kind=kind, n=n, touch5=hits,
                          touch10=sum(e['high'] >= .1 for e in local),
                          close_net5=sum(e['close_net'] >= .05 for e in local),
                          down3=sum(e['low'] <= -.03 for e in local),
                          mean_close_net=np.mean([e['close_net'] for e in local]) if n else None,
                          touch5_wilson95_descriptive=m.wilson(hits, n), current_probability=None,
                          warning='reused research sample, survivor pool, regularclose proxy not actual overnight fill; no untouched calibration'))
m.save(P / 'historical-price-only-stats.json', stats)
m.save(P / 'historical-price-only-events.json', events)

checks = {}
columns = ['dd_high', 'dd10', 'rv90', 'weekly_mid', 'weekly_vol', 'ma200', 'rsi2', 'atr14', 'vol_ratio20']
for symbol in ['IREN', 'APLD', 'COHR', 'AAOI', 'CIFR']:
    d = market[symbol]
    cut = d.index[-25]
    prefix = m.light(d.loc[:cut], market['SMH'].loc[:cut], cal)
    checks[symbol] = bool(np.allclose(prefix.loc[cut, columns].astype(float),
                                    factors[symbol].loc[cut, columns].astype(float), equal_nan=True))
hashes = {}
for folder in ['daily-raw', 'live-raw']:
    manifest = json.loads((P / folder / 'manifest.json').read_text())
    hashes[folder] = all(hashlib.sha256((P / folder / a['file']).read_bytes()).hexdigest() == a['sha256']
                         for a in manifest if 'file' in a)
assert all(checks.values()) and all(hashes.values())
assert cal.next_session(last) == pd.Timestamp('2026-10-05')
# 独立数值校验：代表股票的日收益和量比与原行情重算一致。
manual = {}
for symbol in ['IREN', 'APLD', 'AAOI']:
    d = market[symbol]
    r = next(r for r in rows if r['ticker'] == symbol)
    manual[symbol] = bool(np.isclose(r['vol_ratio20'], d.volume.iloc[-1] / d.volume.iloc[-21:-1].mean())
                          and np.isclose(r['day_return'], d.adjclose.iloc[-1] / d.adjclose.iloc[-2] - 1))
assert all(manual.values())
m.save(P / 'verification.json', dict(as_of=now, stocks=len(rows), requested=len(m.PROTO['stocks']),
                                    loader_errors=errors, factor_errors=factor_errors,
                                    last_completed=last, next_session=cal.next_session(last),
                                    truncation_checks=checks, raw_hashes=hashes, independent_numeric_checks=manual,
                                    incomplete_week_excluded=True, latest_week_is_complete=True,
                                    probability_validation=False, overnight_execution_validation=False))
print('COVERAGE', len(rows), '/', len(m.PROTO['stocks']), 'errors', errors, factor_errors)
for label, rank in [('WASHOUT', washout), ('MOMENTUM', momentum), ('LRCX', lrcx)]:
    print(label)
    for r in rank[:12]:
        print(json.dumps(clean({k: r[k] for k in ['ticker', 'close', 'day_return', 'gate_count', 'dd_high', 'dd10', 'rv90', 'weekly_vol', 'vol_ratio20', 'rsi2', 'atr14', 'last_extended_price', 'vwap5m_proxy', 'reference_entry_cap']})))
print('STRICT', strict)
print('CREDIT', json.dumps(clean(credit)))
