"""Append-only audit of frozen forecasts; no refit on today's rebound."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
old_names = ['2026-09-30-AAOI-W底与93止损', '2026-09-30-AAOI十交易日走势', '2026-09-30-AAOI-AVGO盘中最新因子']
old_dirs = [BASE.parent / name for name in old_names]


def read(p):
    return json.loads(p.read_text())


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in old_dirs for p in folder.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}


def main():
    before = hashes()
    spec = importlib.util.spec_from_file_location('guard', ROOT / 'tools/forecast_evidence.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    original = read(old_dirs[0] / 'decision.json')
    ten = read(old_dirs[1] / 'results.json')
    latest = read(old_dirs[2] / 'results.json')
    r = read(BASE / 'raw/AAOI-live.json')['chart']['result'][0]
    meta = r['meta']
    quote = dt.datetime.fromtimestamp(meta['regularMarketTime'], dt.timezone.utc).astimezone(ZoneInfo('America/New_York'))
    q = pd.DataFrame(r['indicators']['quote'][0])
    q.index = pd.to_datetime(r['timestamp'], unit='s', utc=True).tz_convert('America/New_York')
    cutoff = quote.replace(second=0, microsecond=0)
    q = q.loc[(q.index.date == quote.date()) & (q.index < cutoff) &
              (q.index.hour * 60 + q.index.minute >= 570) &
              (q.index.hour * 60 + q.index.minute < 960)].dropna()
    q = q.loc[q.volume > 0]
    vwap = float((((q.high + q.low + q.close) / 3) * q.volume).sum() / q.volume.sum())
    px = float(meta['regularMarketPrice'])
    findings = []
    evidence = {}
    for symbol in ['AAOI', 'AVGO']:
        f = latest['symbols'][symbol]['forecast']
        rows = f['evaluation_rows']
        errors = np.array([abs(x['ensemble'] - x['actual_log_return']) for x in rows])
        zero_errors = np.array([abs(x['actual_log_return']) for x in rows])
        assert np.isclose(errors.mean(), f['evaluation']['ensemble']['mae_log'])
        assert np.isclose(zero_errors.mean(), f['evaluation']['zero']['mae_log'])
        assert all(x['signal_date'] < x['label_end'] < '2026-09-30' for x in rows)
        inp = {'evaluation': f['evaluation'], 'improvement_ci95': f['mae_improvement_year_block_ci95'],
               'target_scope_validated': False, 'untouched_audit': False,
               'target_scope': 'user intraday left-side entry with absolute stop93 target110' if symbol == 'AAOI' else 'intraday comparison'}
        (BASE / f'{symbol}-evidence-input.json').write_text(json.dumps(inp, indent=2))
        evidence[symbol] = guard.assess_forecast(inp['evaluation'], improvement_ci95=inp['improvement_ci95'],
                            target_scope_validated=False, untouched_audit=False)
    spec = importlib.util.spec_from_file_location('old_indicators', old_dirs[0] / 'model.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    daily = old.load('AAOI')
    daily = daily.loc[[str(d) < '2026-09-30' for d in daily.index]]
    names = ten['protocol']['features']
    corr = old.indicators(daily)[names].dropna().corr()
    corr.to_csv(BASE / 'historical-feature-correlations.csv')
    pairs = [{'a': a, 'b': b, 'correlation': float(corr.loc[a, b])}
             for i, a in enumerate(names) for b in names[i + 1:] if abs(corr.loc[a, b]) >= .65]
    f = latest['symbols']['AAOI']['forecast']
    neighbor_years = sorted({n['signal_date'][:4] for n in f['neighbors']})
    assert '2026' not in neighbor_years
    findings = [
        {'id': 'decision_mapping', 'finding': 'Model without reliable edge still led a negative entry judgment.',
         'correction': 'Set model interpretation weight to zero; no-edge never sets buy=false or market=bearish.'},
        {'id': 'horizon_and_execution', 'finding': 'Daily close/next-open labels do not validate intraday left-side entry; ten-session outcome ends Oct14.',
         'correction': 'Separate entry risk plan from terminal forecast. Intraday outcome is recorded separately, not as final ten-session win/loss.'},
        {'id': 'feature_scope', 'finding': 'Seven fitted factors are price-derived; no explicit W geometry or intraday VWAP/flow. Latest macro/options were diagnostics outside the fitted forecast.',
         'correction': 'Disclose each factor role and observation time. Do not imply diagnostics update model coefficients or probabilities.'},
        {'id': 'strategy_scope', 'finding': 'W confirmation and LRCX trend conditions received excessive influence on a user-defined left-side plan. Neckline110.74 is above user target110.',
         'correction': 'Unconfirmed W is a structure status; do not require neckline breakout or LRCX conditions as universal left-side entry gates.'},
        {'id': 'threshold_ownership', 'finding': 'RR>=2.5 was analyst-selected, not user-specified or backtested. Missing global opportunity score does not reject this specific trade.',
         'correction': 'Show RR as arithmetic; label analyst preferences and restrict certification gates to claims they validate.'},
        {'id': 'regime_and_dependence', 'finding': 'All32 current neighbors predate2026; several price features are correlated. These are limitations, not proven causal attributions for today.',
         'correction': 'No retrospective factor reweighting on one rebound; new structure/flow model requires frozen objectives, incremental tests and future independent data.'},
    ]
    result = {
        'audited_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'quote': {'price': px, 'observed_at': quote.isoformat(), 'source': read(BASE / 'fetch-manifest.json')['source'],
                  'regular_bar_approx_vwap': vwap, 'last_completed_bar': q.index[-1].isoformat(),
                  'return_from_original_model_reference': px / original['model_reference_price'] - 1},
        'original_plan_geometry': guard.trade_geometry(original['model_reference_price'], 93, 110),
        'original_final_quote_geometry': guard.trade_geometry(original['reference_price'], 93, 110),
        'current_geometry': guard.trade_geometry(px, 93, 110),
        'corrected_original_assessment': {
            'strategy': 'unconfirmed_W_left_side_risk_bounded_trial',
            'model_buy_veto': False, 'statistically_certified_buy': False,
            'normal_position_assessment': None, 'position_size': None,
            'reason': 'At about97.5 user plan can be assessed as a conditional left-side trial, with93 invalidation and110 exit. No validated evidence supported a strong no-buy conclusion; positive expectation and risk budget remain unknown.',
            'w_confirmed': False, 'neckline': 110.74, 'right_bottom_low': 93.62,
            'stop_noise_risk_retained': True,
        },
        'model_evidence': evidence, 'findings': findings,
        'feature_correlation_diagnostic': pairs, 'current_neighbor_years': neighbor_years,
        'factor_roles': {'fitted': names, 'intraday_diagnostic': ['VWAP', 'price recovery', 'relative strength', 'volume'],
                         'background_not_fitted': ['macro', 'options IV', 'semiconductor site']},
        'user_report': {'literal_entry_text': '07', 'interpretation_confirmed': False,
                        'claims_missed_purchase_due_to_advice': True, 'actual_fill_verified': False,
                        'shares': None, 'realized_or_lost_profit': None},
        'ten_session_outcome': {'terminal_date': '2026-10-14', 'mature': False},
        'no_orders_executed': True,
    }
    assert hashes() == before
    result['verification'] = {'historical_files_unchanged': True, 'historical_file_count': len(before),
                              'saved_metrics_independently_recomputed': True, 'no_today_refit': True,
                              'completed_minutes_only_vwap': True, 'no_verified_trading_edge_created': True}
    (BASE / 'historical-files-sha256.json').write_text(json.dumps(before, ensure_ascii=False, indent=2))
    (BASE / 'audit-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'quote': result['quote'], 'verification': result['verification'],
                      'strong_feature_correlations': pairs, 'evidence': evidence}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
