"""Read-only diagnosis of the existing gates, after account rules have run.

No result from this module is consumed by an action or allocation rule. The
failure names are the existing engine reasons; missing evidence is reported
only by the same evidence readers used by decide, never by a new rejection.
"""
from .inputs import (
    EvidenceError, eps_revision, finite, member, price_metrics, prices,
)


def closest_candidate(account, facts, asof, *, regime, macro, calendars, clock,
                      contexts, computed, cores, satellites, metal_symbols,
                      max_industry_names, satellite_cap, book):
    """Return the lexicographically nearest ledger asset, without changing it.

    rank_score is an ascending lexicographic tuple: number of failed gates,
    CORE priority, negative frozen dip unit (CORE) or RSI (other roles), then
    new-low priority (other roles only). Unknown measurements remain null in
    JSON and sort last internally. Symbol is the final deterministic tie break.
    """
    executable = {row['symbol'] for row in cores + satellites
                  if row['status'] == '可执行'}
    satellite_rows = {row['symbol']: row for row in satellites}
    held = {row['symbol']: row for row in account['held']}
    candidates = []
    for asset in account['assets']:
        symbol, market, role = (asset.get(key) for key in ('symbol', 'market', 'role'))
        industry = asset.get('industry')
        failed = set(account['reasons'])
        cal = calendars.get(market, {})
        if not macro['data_valid'] or not calendars['US']['open']:
            failed.add('required_macro_evidence_or_us_session_missing')
        if regime == 'UNKNOWN':
            failed.add('official_regime_unknown')
        if macro['pressure'] and account['gross'] is not None and account['gross'] > 1:
            failed.add('confirmed_pressure_gross_gt_1')
        if role == 'CORE' and regime == 'FIRST_CUT':
            failed.add('first_cut_equity_exit_window_metals_hold')
        if role == 'CORE' and regime == 'EASING':
            failed.add('easing_no_new_equity_core')

        supported = (role in {'CORE', 'SATELLITE', 'BALLAST'} and market in {'US', 'CN'}
                     and bool(symbol) and bool(industry))
        if not supported:
            failed.add('unsupported_book_asset')
        metrics = computed.get(symbol)
        if supported:
            if market == 'CN' and (role != 'BALLAST' or industry != 'SOE_DIVIDEND'):
                failed.add('a_share_ballast_only')
            if not cal.get('verified'):
                failed.add('exchange_calendar_unverified')
            elif not cal['open']:
                failed.add('exchange_closed_next_open=' + str(cal['next_open']))
            if cal.get('verified') and cal.get('open'):
                if market == 'CN' and not member(facts, 'CSI300', symbol, asof, 'CN'):
                    failed.add('not_current_csi300_member')
                if role == 'CORE':
                    if len(account['cores']) != 1:
                        failed.add('multiple_global_cores')
                    if market != 'US' or not member(facts, 'SMH', symbol, asof):
                        failed.add('not_current_smh_member')
                elif symbol not in {'SLV', 'SILJ', 'TLT'} and industry not in {'UTILITY', 'SOE_DIVIDEND'}:
                    failed.add('not_rate_suppressed_satellite_bucket')
                if metrics is None:
                    try:
                        metrics = price_metrics(prices(facts, symbol, asof, market))
                    except EvidenceError as error:
                        failed.add(str(error))

            if role == 'CORE':
                if metrics is not None:
                    try:
                        eps_revision(facts, symbol, metrics['window_start'], asof)
                    except EvidenceError as error:
                        failed.add(str(error))
                    if metrics['ret_5d'] > .08:
                        failed.add('five_day_gain_gt_8pct')
                    if metrics['ret_1d'] >= 0 or metrics['down_candle'] is False:
                        failed.add('core_not_down')
                    if metrics['down_candle'] is None:
                        failed.add('candle_open_missing')
                    if metrics['dip_unit'] == 0:
                        failed.add('below_frozen_dip_tier')
                earnings = facts.get('earnings', {}).get(symbol)
                earnings_date = earnings.get('date') if isinstance(earnings, dict) else earnings
                if earnings_date and asof < earnings_date:
                    failed.add('before_earnings_hold_only')
            else:
                if symbol in metal_symbols and clock != 'DIP_ALLOWED':
                    failed.add('metal_clock_' + clock)
                if macro['yield_stable'] is True and symbol not in held:
                    failed.add('yield_stable_no_new_satellites')
                if regime not in {'HIKING', 'PAUSE'}:
                    failed.add('policy_no_new_satellites')
                if metrics is not None:
                    if metrics['at_252_high'] is None:
                        failed.add('prior_252_price_history_missing')
                    elif symbol not in held and metrics['at_252_high']:
                        failed.add('new_satellite_at_252_high')
                    if not ((metrics['rsi14'] is not None and metrics['rsi14'] < 30)
                            or metrics['new_low20'] is True):
                        failed.add('satellite_not_oversold')
                if account['budget_ready'] and account['gross'] is not None:
                    existing = abs(held.get(symbol, {}).get('nominal', 0)) / book['nav']
                    room = max(0., 1 - account['gross']) if macro['pressure'] else book['budget'] / book['nav']
                    assigned = satellite_rows.get(symbol, {}).get('suggested_nominal_cap')
                    if (min(satellite_cap - existing, room, book['budget'] / book['nav']) <= 0
                            or assigned is not None and assigned <= 0):
                        failed.add('satellite_nominal_or_gross_cap_no_room')

            # Use the exact counts seen by the existing sequential ledger loop.
            # This does not reserve another slot or reorder that loop.
            context = contexts[symbol]
            if symbol not in held:
                if context['industry_names'] >= max_industry_names:
                    failed.add('sixth_industry_name')
                if context['names'] >= account['max_names']:
                    failed.add('total_name_cap')

        is_core = role == 'CORE'
        measure = None
        new_low_priority = 0 if is_core else None
        if metrics is not None:
            raw_measure = metrics['dip_unit'] if is_core else metrics['rsi14']
            if finite(raw_measure):
                measure = -raw_measure if is_core else raw_measure
            if not is_core and metrics['new_low20'] is not None:
                new_low_priority = 0 if metrics['new_low20'] else 1
        score = [len(failed), 0 if is_core else 1, measure, new_low_priority]
        sort_key = (*score[:2], float('inf') if measure is None else measure,
                    float('inf') if new_low_priority is None else new_low_priority,
                    symbol or '')
        candidates.append((sort_key, {
            'symbol': symbol, 'market': market, 'role': role,
            'passed': symbol in executable,
            'failed_gates': sorted(failed), 'rank_score': score,
        }))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None
