"""Raw, dated evidence; derived caller scores and returns are never consumed."""
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
import math

import exchange_calendars as xc
import pandas as pd

from macro_regime.factors import rsi, dip_units


class EvidenceError(ValueError):
    pass


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@lru_cache(maxsize=32)
def nyse(year, start_year=None):
    start_year = year - 3 if start_year is None else start_year
    return xc.get_calendar('XNYS', start=f'{start_year}-01-01', end=f'{year+1}-12-31')


def close_time(day, market='US'):
    if market == 'US':
        cal = nyse(pd.Timestamp(day).year)
        if not cal.is_session(day):
            raise EvidenceError('exchange_closed')
        return cal.session_close(day).to_pydatetime()
    return pd.Timestamp(day+'T15:00:00+08:00').to_pydatetime()


def known(record, day, market='US'):
    source = record.get('source')
    if not isinstance(source, str) or not source.strip() or record.get('verified') is not True:
        return False
    try:
        available = datetime.fromisoformat(record['available_at'])
        # Discard future backfills before constructing their historical calendar.
        if (available.tzinfo is None
                or available.astimezone(timezone.utc).date() > datetime.fromisoformat(day).date()):
            return False
        return available <= close_time(day, market)
    except (KeyError, TypeError, ValueError):
        return False


def prices(facts, symbol, asof, market='US'):
    """Each row is a PIT adjusted close, not a latest-history retrofit."""
    records = facts.get('prices', {}).get(symbol, [])
    eligible = []
    for row in records:
        day = row.get('date', '')
        if not day or day > asof:
            continue
        if row.get('closed') is not True or not known(row, day, market):
            continue
        if not finite(row.get('adj_close')) or row['adj_close'] <= 0:
            raise EvidenceError('invalid_adjusted_close')
        eligible.append((day, float(row['adj_close'])))
    eligible.sort()
    if len({day for day, _ in eligible}) != len(eligible):
        raise EvidenceError('duplicate_closed_prices')
    if not eligible or eligible[-1][0] != asof:
        raise EvidenceError('closed_price_history_missing')
    s = pd.Series([v for _, v in eligible], index=pd.DatetimeIndex([d for d, _ in eligible]))
    current = [r for r in records if r.get('date')==asof and r.get('closed') is True and known(r,asof,market)]
    opened = current[-1].get('adj_open')
    s.attrs['adj_open'] = float(opened) if finite(opened) and opened>0 else None
    if market == 'US':
        expected = nyse(pd.Timestamp(asof).year, s.index[0].year).sessions_in_range(s.index[0], asof)
        if not s.index.equals(expected):
            raise EvidenceError('price_session_gap')
    else:
        calendar = facts.get('calendars', {}).get('CN', {})
        expected = [d for d in calendar.get('sessions', []) if str(s.index[0].date()) <= d <= asof]
        if list(s.index.strftime('%Y-%m-%d')) != expected:
            raise EvidenceError('price_session_gap')
    return s


def price_metrics(series):
    if len(series) < 6:
        raise EvidenceError('price_return_history_missing')
    # Exact decimal ratios avoid changing the frozen boundaries by rounding.
    end=Decimal(str(series.iloc[-1]))
    r1 = float(end/Decimal(str(series.iloc[-2]))-1)
    r5 = float(end/Decimal(str(series.iloc[-6]))-1)
    opened=series.attrs.get('adj_open')
    return {'ret_1d':r1, 'ret_5d':r5, 'dip_unit':dip_units(r1,r5),
            'down_candle':None if opened is None else bool(series.iloc[-1]<opened),
            'rsi14':None if len(series)<15 else float(rsi(series).iloc[-1]),
            'new_low20':None if len(series)<21 else bool(series.iloc[-1]<series.iloc[-21:-1].min()),
            'at_252_high':None if len(series)<253 else bool(series.iloc[-1]>=series.iloc[-253:-1].max()),
            'window_start':None if len(series)<63 else series.index[-63].date().isoformat()}


def member(facts, universe, symbol, asof, market='US'):
    rows = [r for r in facts.get('memberships', {}).get(universe, [])
            if r.get('symbol')==symbol and r.get('date') and r['date'] <= asof and known(r, asof, market)]
    rows.sort(key=lambda r:(r['date'],r['available_at']))
    if not rows:
        return False
    final_date = rows[-1]['date']
    decisions = {r.get('member') for r in rows if r['date']==final_date}
    return decisions == {True}


def eps_revision(facts, symbol, window_start, asof):
    if window_start is None:
        raise EvidenceError('consensus_window_missing')
    valid=[]
    for row in facts.get('consensus',{}).get(symbol,[]):
        if not row.get('date') or row['date'] > asof or not known(row,asof):
            continue
        if (not row.get('fiscal_period') or row.get('is_consensus') is not True
                or not finite(row.get('eps'))):
            continue
        valid.append(row)
    if not valid:
        raise EvidenceError('consensus_missing')
    current=max(valid,key=lambda r:(r['date'],r['available_at']))
    baseline=[r for r in valid if r['date']<=window_start and known(r,window_start)
              and r['fiscal_period']==current['fiscal_period']]
    if not baseline:
        raise EvidenceError('same_fiscal_consensus_baseline_missing')
    start=max(baseline,key=lambda r:(r['date'],r['available_at']))
    if start['eps']<=0:
        raise EvidenceError('consensus_nonpositive_baseline')
    exact_revision=Decimal(str(current['eps']))/Decimal(str(start['eps']))-1
    revision=float(exact_revision)
    if exact_revision < Decimal('-.05'):
        raise EvidenceError('eps_downgrade_gt_5pct')
    return {'fiscal_period':current['fiscal_period'],'revision':revision,
            'baseline_eps':start['eps'],'current_eps':current['eps'],
            'baseline_available_at':start['available_at'],'available_at':current['available_at']}


def calendar_state(facts, market, asof):
    if market=='US':
        cal=nyse(pd.Timestamp(asof).year)
        opened=bool(cal.is_session(asof))
        next_open=asof if opened else cal.date_to_session(asof,direction='next').date().isoformat()
        return {'verified':True,'open':opened,'next_open':next_open,'source':'exchange_calendars.XNYS'}
    if market!='CN':
        return {'verified':False,'open':False,'next_open':None,'source':None}
    data=facts.get('calendars',{}).get('CN',{})
    sessions=data.get('sessions',[])
    valid=(data.get('verified') is True and bool(data.get('source'))
           and sessions==sorted(set(sessions)))
    if not valid:
        return {'verified':False,'open':False,'next_open':None,'source':data.get('source')}
    opened=asof in sessions
    after=[d for d in sessions if d>=asof]
    return {'verified':True,'open':opened,'next_open':after[0] if after else None,'source':data['source']}
