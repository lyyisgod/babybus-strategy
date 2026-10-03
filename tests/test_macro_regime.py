"""冻结用户动作预期；合成日线仅验证实现，不验证收益。"""
from dataclasses import replace
from datetime import date
import json

import numpy as np
import pandas as pd
import pytest

from macro_regime.config import AUTHOR_LEVELS, STATISTICS, trading_days_since
from macro_regime.factors import DataError, dip_units, hike_beta, leap_history, numeric_factors, rsi
from macro_regime.state import DecisionInputs, FactorInputs, decide, decide_factors, evaluate


def frame(end="2026-09-30", periods=1325):
    index = pd.bdate_range(end=end, periods=periods)
    t = np.arange(periods)
    result = pd.DataFrame(index=index)
    for s, value in {"IXG": 150, "JNK": 100, "QQQ": 100, "SMH": 100, "SPY": 100,
                     "SLV": 50, "TLT": 100}.items():
        result[s] = result[f"{s}_adj"] = float(value)
    result["DGS10"] = 4 + .02 * np.sin(t / 9) + .00001 * t
    result["DGS2"] = 4.2
    result["DFF"] = 4.3
    result["BAMLH0A0HYM2"] = 3.0
    result["MOVE"] = 100 + np.sin(t / 7)
    result.loc[result.index[-12:], "MOVE"] = 100.0
    result["move_proxy"] = False
    result["VIX"] = 20 + np.sin(t / 5)
    result["TNX"] = 40.0
    return result


def prices(data, symbol, values):
    for column in (symbol, f"{symbol}_adj"):
        data.loc[data.index[-len(values):], column] = np.array(values, dtype=float)


def scenario(name):
    data = frame()
    if name == "leap":
        prices(data, "QQQ", [97.6])
        prices(data, "SMH", [98])
    elif name == "stable":
        t = np.arange(45)
        data.loc[data.index[-45:], "DGS10"] = 4 + .0001 * np.sin(t / 3)
    else:
        raise AssertionError(name)
    return data, "HIKING", .8


FACTOR_FIELDS = (
    "author_expired", "move_proxy", "ixg_below_125", "move_above_125",
    "ixg_spy_below_ma200", "move_pct", "gross", "regime", "jnk_ret_5d",
    "qqq_ret_5d", "smh_ret_5d", "oas_chg_5d", "dgs10_up_5d", "yield_stable",
    "core_ret_1d", "core_ret_5d", "jnk_down_streak", "vix_high", "pair_prev", "trigger_on",
)


# 每行完整指定因子及昨日pair_prev、昨日确认状态trigger_on。
# expected=(action,allow_margin,author_combo,stat_combo,size_unit)。
@pytest.mark.parametrize("name,values,expected", [
    ('author_only_hold', (False, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, True, False), ('CORE_HOLD_NO_MARGIN', False, True, False, 0)),
    ('author_only_delever', (False, False, True, True, False, 80, 1.4, 'HIKING', 0, 0, 0, 0, False, False, -0.04, -0.04, 0, False, True, False), ('DELEVER_TO_1X', False, True, False, 0)),
    ('both_combos', (False, False, True, True, True, 95, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, True, False), ('CORE_HOLD_NO_MARGIN', False, True, True, 0)),
    ('stat_only', (False, False, False, True, True, 95, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, True, False), ('CORE_HOLD_NO_MARGIN', False, False, True, 0)),
    ('no_cross_absolute_ixg', (False, False, True, False, False, 95, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.02, -0.02, 0, False, False, False), ('DIP_ADD', True, False, False, 0.25)),
    ('no_cross_absolute_move', (False, False, False, True, True, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.02, -0.02, 0, False, False, False), ('DIP_ADD', True, False, False, 0.25)),
    ('expired_stat_only', (True, False, True, True, True, 95, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, True, False), ('CORE_HOLD_NO_MARGIN', False, False, True, 0)),
    ('expired_author_off', (True, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, False, False), ('DIP_ADD', True, False, False, 0)),
    ('proxy_stat_only', (False, True, True, True, True, 95, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, True, False), ('CORE_HOLD_NO_MARGIN', False, False, True, 0)),
    ('proxy_no_absolute', (False, True, True, True, False, 95, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, False, False), ('DIP_ADD', True, False, False, 0)),
    ('missing_percentile_author_works', (False, False, True, True, False, None, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, True, False), ('CORE_HOLD_NO_MARGIN', False, True, False, 0)),
    ('missing_percentile_stat_off', (True, False, False, False, True, None, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, False, False), ('DIP_ADD', True, False, False, 0)),
    ('exact_80_not_high', (True, False, False, False, True, 80, 0.8, 'HIKING', 0, 0, 0, 0, False, False, 0, 0, 0, False, False, False), ('DIP_ADD', True, False, False, 0)),
    ('jnk_seven_down', (False, False, False, False, False, 50, 0.8, 'HIKING', -0.009, 0, 0, 0, False, False, -0.02, -0.02, 7, False, False, False), ('BUY_CORE_SEMI', True, False, False, 0.25)),
    ('rate_not_credit', (False, False, False, False, False, 50, 0.8, 'HIKING', -0.02, 0, 0, 0, True, False, -0.02, -0.02, 0, False, False, False), ('BUY_CORE_SEMI', True, False, False, 0.25)),
    ('confirmed_selloff', (False, False, True, True, False, 50, 0.8, 'HIKING', -0.03, -0.03, -0.03, 0.1, True, False, -0.04, -0.04, 0, False, True, False), ('DELEVER_TO_1X', False, True, False, 0)),
    ('risk_on', (False, False, False, False, False, 50, 0.8, 'HIKING', 0.01, -0.01, 0, 0, False, False, -0.02, -0.02, 0, False, False, False), ('INDEX_DIP_BUY', True, False, False, 0.25)),
    ('first_cut_prices_high', (False, False, False, False, False, 50, 0.8, 'FIRST_CUT', 0, 0, 0, 0, False, False, -0.1, -0.1, 0, False, False, False), ('CORE_EXIT_WINDOW', True, False, False, 0)),
    ('first_cut_pressure', (False, False, True, True, False, 50, 1.4, 'FIRST_CUT', -0.03, -0.03, -0.03, 0.1, True, True, -0.1, -0.1, 0, False, True, False), ('CORE_EXIT_WINDOW', False, True, False, 0)),
    ('yield_stable', (False, False, False, False, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, True, -0.02, -0.02, 0, False, False, False), ('EARNINGS_SLEEVE', True, False, False, 0)),
    ('watch_stable_same_action', (False, False, True, False, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, True, -0.02, -0.02, 0, False, False, False), ('EARNINGS_SLEEVE', True, False, False, 0)),
    ('chase_block', (False, False, False, False, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.04, 0.1, 0, False, False, False), ('DIP_ADD', True, False, False, 0)),
    ('pause', (False, False, False, False, False, 50, 0.8, 'PAUSE', 0, 0, 0, 0, False, False, -0.04, -0.04, 0, False, False, False), ('CORE_HOLD', True, False, False, 0)),
    ('easing_hold', (False, False, False, False, False, 50, 0.8, 'EASING', 0, 0, 0, 0, False, True, -0.04, -0.04, 0, False, False, False), ('HOLD', True, False, False, 0)),
    ('easing_reentry', (False, False, False, False, False, 50, 0.8, 'EASING', 0, 0, 0, 0, False, False, -0.04, -0.04, 0, True, False, False), ('REENTRY_WATCH', True, False, False, 0)),
    ('core_add_quarter', (False, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.02, -0.02, 0, False, True, False), ('CORE_ADD_NO_MARGIN', False, True, False, 0.25)),
    ('core_add_half', (False, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.04, -0.04, 0, False, True, False), ('CORE_ADD_NO_MARGIN', False, True, False, 0.5)),
    ('core_add_one', (False, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.07, -0.07, 0, False, True, False), ('CORE_ADD_NO_MARGIN', False, True, False, 1)),
    ('stress_chase_hold', (False, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.04, 0.1, 0, False, True, False), ('CORE_HOLD_NO_MARGIN', False, True, False, 0)),
    ('first_close_watch', (False, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.02, -0.02, 0, False, False, False), ('DIP_ADD', True, True, False, 0.25)),
    ('second_close_trigger', (False, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.02, -0.02, 0, False, True, False), ('CORE_ADD_NO_MARGIN', False, True, False, 0.25)),
    ('break_before_confirm', (False, False, False, False, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.02, -0.02, 0, False, True, False), ('DIP_ADD', True, False, False, 0.25)),
    ('clear_first_close_keeps_trigger', (False, False, False, False, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.02, -0.02, 0, False, True, True), ('CORE_ADD_NO_MARGIN', False, False, False, 0.25)),
    ('clear_second_close_releases', (False, False, False, False, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.02, -0.02, 0, False, False, True), ('DIP_ADD', True, False, False, 0.25)),
    ('relapse_after_one_clear', (False, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.02, -0.02, 0, False, False, True), ('CORE_ADD_NO_MARGIN', False, True, False, 0.25)),
    ('clear_second_close_watch_label', (False, False, True, False, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, True, -0.02, -0.02, 0, False, False, True), ('EARNINGS_SLEEVE', True, False, False, 0)),
    ('confirmed_still_on', (False, False, True, True, False, 50, 0.8, 'HIKING', 0, 0, 0, 0, False, False, -0.04, -0.04, 0, False, True, True), ('CORE_ADD_NO_MARGIN', False, True, False, 0.5)),
])
def test_user_action_table(name, values, expected):
    assert len(values) == len(FACTOR_FIELDS)
    inputs = FactorInputs(**dict(zip(FACTOR_FIELDS, values)))
    result = decide_factors(inputs)
    assert (result["action"], result["constraints"]["allow_margin"],
            result["author_combo"], result["stat_combo"], result["size_unit"]) == expected
    assert result["size_unit"] in (0, .25, .5, 1)
    assert result["pair_prev"] is inputs.pair_prev
    assert result["trigger_on"] == (result["stress"] == "TRIGGER")
    if result["action"] in ("DELEVER_TO_1X", "CORE_EXIT_WINDOW", "HOLD", "EARNINGS_SLEEVE", "CORE_HOLD_NO_MARGIN"):
        assert result["size_unit"] == 0
    if result["action"] == "CORE_ADD_NO_MARGIN":
        assert result["size_unit"] > 0
        assert result["constraints"]["allow_margin"] is False
        assert result["constraints"]["max_gross"] == 1
    if name == "clear_first_close_keeps_trigger":
        assert result["stress"] == "TRIGGER" and result["clear_streak"] == 1
    if name in ("clear_second_close_releases", "clear_second_close_watch_label"):
        assert result["stress"] != "TRIGGER" and result["clear_streak"] == 2
    if name == "first_close_watch":
        assert result["stress"] == "WATCH" and result["pair_streak"] == 1
    if name == "second_close_trigger":
        assert result["stress"] == "TRIGGER" and result["pair_streak"] == 2
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("daily,five,expected", [
    (.01, 0, 0), (0, 0, 0), (-.0099, 0, 0), (-.01, 0, 0),
    (-.0299, 0, .25), (-.03, 0, .25), (-.0599, 0, .5), (-.06, 0, .5),
    (-.0601, 0, 1), (-.02, .08, .25), (-.02, .08001, 0),
])
def test_dip_boundaries(daily, five, expected):
    assert dip_units(daily, five) == expected


@pytest.mark.parametrize("policy,stress,div,gross,stable,vix,action", [
    ("FIRST_CUT", "TRIGGER", "CONFIRMED_SELLOFF", 1.4, True, True, "CORE_EXIT_WINDOW"),
    ("HIKING", "TRIGGER", "CONFIRMED_SELLOFF", .8, True, False, "DELEVER_TO_1X"),
    ("HIKING", "TRIGGER", "UNCONFIRMED_CREDIT", .8, True, False, "CORE_ADD_NO_MARGIN"),
    ("HIKING", "NONE", "RATE_NOT_CREDIT", .8, True, False, "BUY_CORE_SEMI"),
    ("HIKING", "NONE", "RISK_ON_DIVERGENCE", .8, True, False, "INDEX_DIP_BUY"),
    ("HIKING", "NONE", "NONE", .8, True, False, "EARNINGS_SLEEVE"),
    ("HIKING", "WATCH", "NONE", .8, True, False, "EARNINGS_SLEEVE"),
    ("PAUSE", "NONE", "NONE", .8, True, False, "CORE_HOLD"),
    ("PAUSE", "TRIGGER", "NONE", 1.4, False, False, "DELEVER_TO_1X"),
    ("EASING", "NONE", "NONE", .8, True, True, "REENTRY_WATCH"),
    ("EASING", "NONE", "NONE", .8, True, False, "HOLD"),
])
def test_priority_table(policy, stress, div, gross, stable, vix, action):
    assert decide(DecisionInputs(policy, stress, div, gross, .5, stable, vix))["action"] == action


@pytest.mark.parametrize("proposed_unit", [0, .25, .5, 1])
@pytest.mark.parametrize("policy,stress,div,gross,stable,expected", [
    ("HIKING", "TRIGGER", "NONE", 1.4, False, "DELEVER_TO_1X"),
    ("FIRST_CUT", "TRIGGER", "CONFIRMED_SELLOFF", 1.4, False, "CORE_EXIT_WINDOW"),
    ("EASING", "NONE", "NONE", .8, False, "HOLD"),
    ("HIKING", "NONE", "NONE", .8, True, "EARNINGS_SLEEVE"),
])
def test_zero_size_actions_suppress_every_proposed_unit(proposed_unit, policy, stress, div, gross, stable, expected):
    result = decide(DecisionInputs(policy, stress, div, gross, proposed_unit, stable))
    assert result["action"] == expected
    assert result["size_unit"] == 0


def test_two_day_confirmation_and_release_sequence_with_explicit_state():
    timeline = [
        # 当日两维、昨日pair、昨日确认状态、当天状态及动作。
        (True, True, False, False, "WATCH", "DIP_ADD"),
        (True, True, True, False, "TRIGGER", "CORE_ADD_NO_MARGIN"),
        (False, False, True, True, "TRIGGER", "CORE_ADD_NO_MARGIN"),
        (False, False, False, True, "NONE", "DIP_ADD"),
        (True, True, False, False, "WATCH", "DIP_ADD"),
        (True, True, True, False, "TRIGGER", "CORE_ADD_NO_MARGIN"),
    ]
    history, previous_result = [], None
    for ixg_weak, move_high, pair_prev, trigger_on, expected_stress, expected_action in timeline:
        inputs = FactorInputs(False, False, ixg_weak, move_high, False, 50, .8, "HIKING",
                              0, 0, 0, 0, False, False, -.02, -.02,
                              pair_prev=pair_prev, trigger_on=trigger_on)
        if previous_result is not None:
            assert inputs.pair_prev == previous_result["pair"]
            assert inputs.trigger_on == previous_result["trigger_on"]
        result = decide_factors(inputs)
        assert result["stress"] == expected_stress
        assert result["action"] == expected_action
        assert result["size_unit"] == .25
        assert result["constraints"]["allow_margin"] == (expected_stress != "TRIGGER")
        replay = decide_factors(inputs, history=tuple(history))
        assert replay["stress"] == result["stress"]
        assert replay["action"] == result["action"]
        assert replay["size_unit"] == result["size_unit"]
        history.append(inputs)
        previous_result = result


def test_rate_not_credit_and_balance_sheet_are_distinct():
    data = frame()
    prices(data, "JNK", [100, 100, 99, 98.5, 98, 97])
    data.loc[data.index[-6:], "DGS10"] = [4, 4.02, 4.04, 4.06, 4.08, 4.1]
    result = evaluate(data, "HIKING", .8)
    assert result["divergence"] == "RATE_NOT_CREDIT"
    assert result["action"] == "BUY_CORE_SEMI"
    assert not result["balance_sheet_stress"]
    prices(data, "IXG", [124, 124])
    data.loc[data.index[-2:], "MOVE"] = 130
    data.loc[data.index[-1], "BAMLH0A0HYM2"] = 3.2
    assert evaluate(data, "HIKING", .8)["balance_sheet_stress"]


def test_satellite_requires_negative_beta_and_oversold_and_correct_action():
    data = frame()
    # 连续上升利率、反向资产，冻结可解释的负斜率。
    t = np.arange(80)
    data.loc[data.index[-80:], "DGS10"] = 4 + .001 * t + .000003 * t * t
    delta = data.DGS10.diff().tail(79).to_numpy()
    prices(data, "SLV", 50 * np.cumprod(1 - 2 * delta))
    prices(data, "JNK", 100 * .998 ** np.arange(1, 8))
    result = evaluate(data, "HIKING", .8)
    assert result["action"] == "BUY_CORE_SEMI"
    candidate = next(item for item in result["satellite_candidates"] if item["symbol"] == "SLV")
    assert candidate["max_nominal"] == .02
    assert candidate["limit_kind"] == "suggested_nominal_cap"
    assert candidate["position_calculated"] is False
    prices(data, "IXG", [124, 124])
    data.loc[data.index[-2:], "MOVE"] = 130
    assert evaluate(data, "HIKING", .8)["satellite_candidates"] == []


def test_beta_known_ols_and_flat_rsi():
    yields = pd.Series(4 + np.cumsum(.01 * np.sin(np.arange(100))))
    returns = .2 * yields.diff().fillna(0) + .001
    close = 100 * (1 + returns).cumprod()
    assert hike_beta(close, yields).iloc[-1] == pytest.approx(.2, abs=1e-10)
    assert rsi(pd.Series([100.] * 20)).iloc[-1] == 50
    assert rsi(pd.Series(np.arange(1., 21.))).iloc[-1] == 100
    assert rsi(pd.Series(np.arange(21., 1., -1))).iloc[-1] == 0


def test_vix_spike_and_nfp_are_labels_not_sells():
    data = frame()
    data.loc[data.index[-1], "VIX"] = 40
    a = evaluate(data, "HIKING", .8)
    b = evaluate(data, "HIKING", .8, nfp_change=160001)
    assert a["vix_spike"]
    assert a["action"] == b["action"] == "DIP_ADD"


@pytest.mark.parametrize("damage", ["missing_column", "missing_today", "missing_yield_window", "short_history", "zero_yield_variance", "duplicate", "infinity"])
def test_missing_or_invalid_input_fails_closed(damage):
    data = frame()
    if damage == "missing_column":
        data = data.drop(columns="DGS2")
    elif damage == "missing_today":
        data.loc[data.index[-1], "DFF"] = np.nan
    elif damage == "missing_yield_window":
        data.loc[data.index[-20], "DGS10"] = np.nan
    elif damage == "short_history":
        data = data.tail(250)
    elif damage == "zero_yield_variance":
        data.DGS10 = 4.0
    elif damage == "duplicate":
        data = pd.concat([data, data.tail(1)])
    elif damage == "infinity":
        data.loc[data.index[-1], "MOVE"] = np.inf
    with pytest.raises(DataError):
        evaluate(data, "HIKING", .8)


def test_future_prefix_does_not_change_past_result():
    data = frame()
    stop = data.index[-6]
    a = evaluate(data.loc[:stop], "HIKING", .8)
    data.loc[data.index > stop, "MOVE"] = 10000
    data.loc[data.index > stop, ["IXG", "IXG_adj"]] = 1
    b = evaluate(data.loc[:stop], "HIKING", .8)
    assert a == b


def test_author_expiry_63_64_sessions_with_nyse_holidays():
    import exchange_calendars as xcals
    level = AUTHOR_LEVELS["IXG"]
    cal = xcals.get_calendar("XNYS", start="2026-09-30", end="2027-03-01")
    sessions = cal.sessions_in_range("2026-10-01", "2027-03-01")
    day63, day64 = sessions[62].date(), sessions[63].date()
    assert trading_days_since(level.as_of, day63) == 63
    assert trading_days_since(level.as_of, day64) == 64
    assert not level.expired(day63)
    assert level.expired(day64)
    assert level.expired(date(2027, 3, 1))
    # 周末不推进；2026-10-12股市照常交易，不能使用国债或普通工作日日历。
    assert trading_days_since(level.as_of, date(2026, 10, 3)) == 2
    assert trading_days_since(level.as_of, date(2026, 10, 12)) == 8
    inputs = FactorInputs(level.expired(day63), False, True, True, False, 50, .8, "HIKING",
                          0, 0, 0, 0, False, False, 0, 0, pair_prev=True, trigger_on=False)
    result = decide_factors(inputs)
    assert result["author_combo"]
    assert not result["stat_combo"]
    assert result["action"] == "CORE_HOLD_NO_MARGIN"
    assert result["size_unit"] == 0
    next_day = decide_factors(replace(inputs, author_expired=level.expired(day64),
                                     pair_prev=result["pair"], trigger_on=result["trigger_on"]))
    assert not next_day["author_combo"]
    assert next_day["stress"] == "TRIGGER" and next_day["clear_streak"] == 1
    assert next_day["action"] == "CORE_HOLD_NO_MARGIN" and next_day["size_unit"] == 0
    cleared = decide_factors(replace(inputs, author_expired=True,
                                    pair_prev=next_day["pair"], trigger_on=next_day["trigger_on"]))
    assert cleared["stress"] == "NONE" and cleared["clear_streak"] == 2


@pytest.mark.parametrize("regime", ["HIKING", "PAUSE", "FIRST_CUT", "EASING"])
def test_watch_has_no_effect_on_action_or_margin(regime):
    baseline = FactorInputs(False, False, False, False, False, 50, 1.4, regime,
                            0, 0, 0, 0, False, True, -.02, -.02)
    watch = replace(baseline, ixg_below_125=True)
    a = decide_factors(baseline, history=(baseline,))
    b = decide_factors(watch, history=(watch,))
    assert a["stress"] == "NONE" and b["stress"] == "WATCH"
    assert a["action"] == b["action"]
    assert a["constraints"] == b["constraints"]
    assert b["constraints"]["allow_margin"] is True


@pytest.mark.parametrize("move_count", [251, 252, 300])
def test_move_minimum_observations_and_author_independence(move_count):
    data = frame(end="2026-10-02")
    data["IXG"] = 124.  # 复权相对值恒定，统计IXG不弱。
    data.loc[data.index[:-move_count], "MOVE"] = np.nan
    data.loc[data.index[-2:], "MOVE"] = 130
    factors = numeric_factors(data)
    if move_count < 252:
        assert pd.isna(factors.move_pct5y.iloc[-1])
    else:
        assert np.isfinite(factors.move_pct5y.iloc[-1])
    result = evaluate(data, "HIKING", .8)
    assert result["stress"]["author_combo"] is True
    assert result["stress"]["stat_combo"] is False
    assert result["stress"]["state"] == "TRIGGER"
    assert result["constraints"]["allow_margin"] is False
    if move_count < 252:
        assert result["stress"]["move_pct"] is None
    json.dumps(result, allow_nan=False)


def test_proxy_change_rewarms_stat_leg_but_does_not_fake_zero_percentile():
    data = frame()
    data.loc[data.index[-10:], "move_proxy"] = True
    result = evaluate(data, "HIKING", .8)
    assert result["stress"]["move_pct"] is None
    assert result["stress"]["stat_combo"] is False
    assert result["stress"]["author_combo"] is False


def test_leap_cooldown_day20_blocked_day21_allowed_and_weekends_do_not_count():
    import exchange_calendars as xcals
    cal = xcals.get_calendar("XNYS", start="2026-10-01", end="2026-12-01")
    sessions = cal.sessions_in_range("2026-10-01", "2026-12-01")[:23]
    returns = pd.Series(0., index=sessions)
    returns.iloc[[0, 1, 5, 20, 21, 22]] = -.024
    labels = leap_history(returns)
    assert list(np.flatnonzero(labels.to_numpy())) == [0, 21]
    # 顺序回放与重复评估同一完整前缀一致，不依赖运行时全局变量。
    for i in range(1, len(returns)):
        assert leap_history(returns.iloc[:i+1]).iloc[-1] == labels.iloc[i]


def test_leap_cooldown_is_used_by_snapshot():
    data = frame(end="2026-10-02")
    prices(data, "QQQ", [100, 97.6, 97.6, 95.2576])
    result = evaluate(data, "HIKING", .8)
    assert "LEAP_2Y_0_6D" not in result["labels"]
    assert any("20 trading-session cooldown" in reason for reason in result["reason"])


def test_statistical_thresholds_are_frozen():
    assert STATISTICS.relative_ma_days == 200
    assert STATISTICS.move_quantile == .8
    assert STATISTICS.move_window == 1260
    assert STATISTICS.move_min_periods == 252


def test_single_confirming_close_is_watch_and_does_not_disable_margin():
    inputs = FactorInputs(False, False, True, True, False, 50, .8, "HIKING",
                          0, 0, 0, 0, False, False, -.02, -.02)
    result = decide_factors(inputs)
    assert result["author_combo"] and not result["stat_combo"]
    assert result["stress"] == "WATCH"
    assert result["action"] == "DIP_ADD"
    assert result["constraints"]["allow_margin"]


def test_snapshot_first_leap_and_stable_factor_integration():
    data, _, _ = scenario("leap")
    result = evaluate(data, "HIKING", .8)
    assert "LEAP_2Y_0_6D" in result["labels"]
    assert result["action"] == "DIP_ADD" and result["size_unit"] == .25
    data, _, _ = scenario("stable")
    result = evaluate(data, "HIKING", .8)
    assert result["yield_stable"]
    assert result["action"] == "EARNINGS_SLEEVE"
    assert not result["satellite_candidates"]


@pytest.mark.parametrize("name", ["jnk_ret_5d", "qqq_ret_5d", "smh_ret_5d", "oas_chg_5d", "core_ret_1d", "core_ret_5d"])
def test_direct_factor_missing_required_value_fails(name):
    inputs = FactorInputs(False, False, True, True, False, None, .8, "HIKING",
                          0, 0, 0, 0, False, False, 0, 0)
    with pytest.raises(DataError, match="必需因子缺失"):
        decide_factors(replace(inputs, **{name: np.nan}))
