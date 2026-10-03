"""冻结关注池，刷新完整周五日线；研究排序，不训练胜率、不交易。"""
from pathlib import Path
import sys, json, importlib.util
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
sys.path.insert(0, str(ROOT))
from strategies.bb_washout_bounce.data import save, load_config, fetch_market, load_market, calendar
from macro_regime.data import fetch_dataset
from macro_regime.state import evaluate

prior = ROOT / '研究记录/2026-10-01-夜盘Babybus全面筛选/scan.py'
spec = importlib.util.spec_from_file_location('known_price_screen', prior)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
now = datetime.now(timezone.utc).isoformat()
names = json.loads((ROOT / '研究记录/2026-10-02-Babybus次日筛选-1438更新/protocol.json').read_text())['stocks']
cfg = load_config()
cfg['data']['workers'] = 8
context = ['QQQ', 'SMH', 'SPY', 'JNK', 'SLV', 'TLT', 'IXG', '^MOVE', '^VIX', '^TNX', 'TSLL']
save(P / 'protocol.json', {
    'frozen_at': now, 'stocks': names, 'context': context, 'decision_daily_cutoff': '2026-10-02',
    'entry': '2026-10-04 Sunday overnight executable ask unknown',
    'exit_deadline': '2026-10-05T15:55:00-04:00',
    'target': 'intraday gross +5%/+10% from actual fill; realized net exit separate',
    'universe': 'inherited fixed attention pool; not whole market or historical PIT universe',
    'price_gates': 'unchanged prior seven gates and LRCX branch; no parameter fit',
    'policy_regime': 'HIKING', 'policy_source': 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm',
    'policy_basis': 'published Sep16 +25bp decision; analyst classification authorized by user',
    'macro_gross_input': 0, 'macro_gross_note': 'unleveraged scenario only, actual account unknown',
    'model_probability': None, 'macro_paper_N': 0, 'no_orders': True,
})


def macro():
    try:
        dataset = fetch_dataset('2026-10-02')
        result = evaluate(dataset.frame, 'HIKING', 0,
                          dgs10_publications=dataset.dgs10_publications,
                          move_publications=dataset.move_publications)
        result['data'] = dataset.metadata
        result['gross_input_note'] = '0 is research scenario, not user exposure'
        save(P / 'macro_snapshot.json', result)
        print('macro', result['stress']['state'], result['action'], result['yield_stable'], flush=True)
    except Exception as error:
        save(P / 'macro_error.json', {'error': str(error), 'whole_snapshot_unavailable': True})
        print('macro failed', str(error), flush=True)


def daily():
    manifest = fetch_market(names + context, P / 'daily-raw', cfg)
    print('download', sum('error' not in r for r in manifest), '/', len(manifest), flush=True)
    market, metadata, errors, last = load_market(P / 'daily-raw', cfg, now)
    assert str(last.date()) == '2026-10-02'
    rows, factors = [], {}
    for symbol in names:
        try:
            data = market[symbol]
            assert data.index[-1] == last, 'latest complete session missing'
            f = m.light(data, market['SMH'], calendar(cfg))
            factors[symbol] = f
            row = f.iloc[-1].to_dict()
            row.update(ticker=symbol, session=last, price_gates=m.gates(row),
                       high=float(data.high.iloc[-1]), low=float(data.low.iloc[-1]),
                       r5=float(data.adjclose.iloc[-1] / data.adjclose.iloc[-6] - 1),
                       r3=float(data.adjclose.iloc[-1] / data.adjclose.iloc[-4] - 1),
                       ma50=float(data.adjclose.rolling(50).mean().iloc[-1] / data.factor.iloc[-1]),
                       quote_at=pd.Timestamp(metadata[symbol]['provider']['regularMarketTime'], unit='s', tz='UTC'),
                       close_retrace_fraction=float((data.high.iloc[-1]-data.close.iloc[-1])/(data.high.iloc[-1]-data.low.iloc[-1])),
                       volume=float(data.volume.iloc[-1]), source=f'https://finance.yahoo.com/quote/{symbol}/history/')
            row['gate_count'] = sum(row['price_gates'].values())
            row['momentum_branch'] = bool(f.momentum_analog.iloc[-1])
            rows.append(row)
        except Exception as error:
            errors[symbol] = str(error)
    save(P / 'all_factors.json', rows)
    liquid = [r for r in rows if r['adv20'] >= 50e6]
    wash = sorted([r for r in liquid if .35 <= r['dd_high'] <= .65],
                  key=lambda r: (r['gate_count'], r['vol_ratio20']), reverse=True)
    momentum = sorted([r for r in liquid if r['momentum_branch']],
                      key=lambda r: (r['vol_ratio20'], r['relative20']), reverse=True)
    trend = sorted([r for r in liquid if r['lrcx_branch']], key=lambda r: r['relative20'], reverse=True)
    save(P / 'rankings.json', {'washout': wash, 'momentum': momentum, 'lrcx_rebound': trend})
    save(P / 'market_context.json', {s: {'date': d.index[-1], 'close': float(d.close.iloc[-1]),
         'r1': float(d.adjclose.iloc[-1]/d.adjclose.iloc[-2]-1)} for s,d in market.items() if s in context})
    probes = {}
    for symbol in ['IREN', 'APLD', 'LRCX', 'VICR']:
        if symbol not in factors:
            continue
        data = market[symbol]; cut = data.index[-25]
        shortened = m.light(data.loc[:cut], market['SMH'].loc[:cut], calendar(cfg))
        cols = ['dd_high','dd10','rv90','weekly_mid','weekly_vol','ma200','rsi2','atr14','vol_ratio20']
        probes[symbol] = bool(np.allclose(shortened.loc[cut,cols].astype(float),
                                         factors[symbol].loc[cut,cols].astype(float), equal_nan=True))
    save(P / 'verification.json', {'fetched_at': now, 'last_completed': last, 'requested': len(names),
         'factor_count': len(rows), 'errors': errors, 'prefix_checks': probes,
         'macro_version': 'v1.1', 'current_probability': None, 'overnight_orderbook': None})
    assert all(probes.values())
    if 'TSLA' in market:
        d = market['TSLA']; gaps = []
        for i in range(max(1,len(d)-60),len(d)):
            previous, today = d.iloc[i-1], d.iloc[i]
            later = d.iloc[i:]
            if today.open > previous.high and later.low.min() > previous.high:
                gaps.append({'direction':'up','date':d.index[i], 'bottom':float(previous.high),
                             'top':float(min(today.open,later.low.min()))})
            if today.open < previous.low and later.high.max() < previous.low:
                gaps.append({'direction':'down','date':d.index[i], 'bottom':float(max(today.open,later.high.max())),
                             'top':float(previous.low)})
        save(P / 'tsla_user_framework.json', {'gap_definition': 'opening beyond previous full range; only residual unfilled portion',
             'friday_open_vs_previous_close': float(d.open.iloc[-1]/d.close.iloc[-2]-1),
             'gaps_last_60_sessions': gaps, 'TSLL': {'close':float(market['TSLL'].close.iloc[-1]),
             'r1':float(market['TSLL'].adjclose.iloc[-1]/market['TSLL'].adjclose.iloc[-2]-1),
             'quote_at':pd.Timestamp(metadata['TSLL']['provider']['regularMarketTime'],unit='s',tz='UTC'),
             'left_anchor_user_defined':10}})
    for kind, group in [('washout',wash),('momentum',momentum),('lrcx_rebound',trend)]:
        print(kind, flush=True)
        for row in group[:12]:
            print({k:row[k] for k in ['ticker','close','day_return','gate_count','vol_ratio20','dd10','rsi2','ma200','atr14','close_range']}, flush=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(daily), pool.submit(macro)]
        for future in futures:
            future.result()
