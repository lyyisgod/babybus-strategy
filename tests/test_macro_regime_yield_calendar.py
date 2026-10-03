"""v1.1公布日口径、缺口验证及向后映射的防泄露测试。"""
import json

import numpy as np
import pandas as pd
import pytest

from macro_regime.config import STATISTICS, trading_days_since
from macro_regime.factors import DataError, numeric_factors
from macro_regime.state import DecisionInputs, decide, evaluate
from macro_regime.yield_calendar import nyse_sessions, treasury_closed, yield_factors
from test_macro_regime import frame


def stock_frame(end="2026-11-30", periods=800):
    data = frame(end, periods)
    data = data.loc[nyse_sessions(data.index[0], data.index[-1])].copy()
    for day in data.index:
        if treasury_closed(day):
            data.loc[day, "DGS10"] = np.nan
    # 两条腿最终均稳定；仅为功能测试，不是回测观察。
    pubs = data.DGS10.dropna().index[-45:]
    data.loc[pubs, "DGS10"] = 4 + .0001 * np.sin(np.arange(45) / 3)
    data.loc[data.index[-45:], "MOVE"] = 100
    return data


def test_windows_and_thresholds_remain_frozen():
    assert (STATISTICS.yield_std_days, STATISTICS.yield_median_days,
            STATISTICS.yield_confirm_days, STATISTICS.beta_days) == (10, 252, 5, 60)
    assert (STATISTICS.relative_ma_days, STATISTICS.move_quantile) == (200, .8)


@pytest.mark.parametrize("holiday", ["2025-10-13", "2025-11-11", "2026-10-12", "2026-11-11"])
def test_treasury_holiday_carries_completed_factor_not_yield(holiday):
    data = stock_frame()
    on = pd.Timestamp(holiday)
    factors = numeric_factors(data)
    previous = data.index[data.index.get_loc(on) - 1]
    assert pd.isna(data.loc[on, "DGS10"])
    assert factors.loc[on, "dgs10_status"] == "treasury_closed"
    assert factors.loc[on, "yield_asof"] == previous
    for column in ("yield_std10", "yield_std_median252", "yield_low_streak", "yield_leg"):
        assert factors.loc[on, column] == factors.loc[previous, column]
    assert factors.loc[on:, "yield_std10"].notna().all()
    assert factors.loc[on:, "yield_stable"].notna().all()
    # 60股票交易日回归不因正常国债假期出现伪零或丢失整个窗口。
    assert factors.loc[on, "SMH_beta"] == pytest.approx(0)


def test_publication_delta_and_std_match_dropna_reference():
    data = stock_frame()
    observed = data.DGS10.dropna()
    expected = observed.diff().rolling(10, min_periods=10).std(ddof=1)
    expected_median = expected.rolling(252, min_periods=200).median()
    out = numeric_factors(data)
    pd.testing.assert_series_equal(out.loc[observed.index, "yield_std10"], expected,
                                   check_names=False, check_freq=False)
    pd.testing.assert_series_equal(out.loc[observed.index, "yield_std_median252"], expected_median,
                                   check_names=False, check_freq=False)
    # 哥伦布日后一次公布的差值跨过缺口，不能diff后补零。
    date = pd.Timestamp("2026-10-13")
    assert observed.diff().loc[date] == pytest.approx(observed.loc[date] - observed.loc["2026-10-09"])


@pytest.mark.parametrize("count,known", [(209, False), (210, True), (250, True)])
def test_median_minimum_200_vol_observations_and_snapshot_null(count, known):
    data = stock_frame(end="2026-09-30")
    dates = data.DGS10.dropna().index
    data.loc[dates[:-count], "DGS10"] = np.nan
    result = evaluate(data, "HIKING", .8)
    assert result["version"] == "v1.1"
    assert (result["yield_stable"] is not None) is known
    if not known:
        assert result["action"] != "EARNINGS_SLEEVE"
        assert any("insufficient_yield_history" in r for r in result["reason"])
    json.dumps(result, allow_nan=False)


def test_std_requires_10_publication_changes():
    data = stock_frame().tail(14)
    out = yield_factors(data.index, data.DGS10, data.MOVE)
    publication_vol = out.loc[data.DGS10.dropna().index, "yield_std10"]
    assert publication_vol.iloc[:10].isna().all()
    assert publication_vol.iloc[10:].notna().all()
    assert out.yield_stable.isna().all()


def test_five_day_confirmation_is_independent_on_each_publication_calendar():
    data = stock_frame()
    move = pd.Series(100 + np.arange(len(data)) * .01, index=data.index)
    # 最后5次公布下降；中间股票开市但MOVE没有新的公布。
    last = move.index[-8:]
    move.loc[last] = move.loc[last[0]] - np.arange(1, 9)
    move = move.drop(last[3])
    out = yield_factors(data.index, data.DGS10, move)
    assert out.loc[last[2], "move_nonrise_streak"] == 3
    assert out.loc[last[3], "move_nonrise_streak"] == 3
    assert out.loc[last[3], "move_asof"] == last[2]
    assert not out.loc[last[4], "move_yield_leg"]
    assert out.loc[last[5], "move_yield_leg"]
    assert out.loc[last[5], "yield_stable"]


def test_yield_five_publication_streak_reference_and_holiday_does_not_increment():
    data = stock_frame()
    out = yield_factors(data.index, data.DGS10, data.MOVE)
    observed = data.DGS10.dropna()
    low = out.loc[observed.index, "yield_std10"] < out.loc[observed.index, "yield_std_median252"]
    counts = low.astype(int).groupby((~low).cumsum()).cumsum()
    pd.testing.assert_series_equal(out.loc[observed.index, "yield_low_streak"], counts,
                                   check_names=False, check_dtype=False, check_freq=False)
    known = out.loc[observed.index, "yield_std_median252"].notna()
    assert (out.loc[observed.index, "yield_leg"][known] == (counts[known] >= 5)).all()
    assert out.loc["2026-11-11", "yield_low_streak"] == out.loc["2026-11-10", "yield_low_streak"]


@pytest.mark.parametrize("date", ["2026-10-13", "2026-11-27"])
def test_unexpected_gap_including_early_close_errors_whole_snapshot(date):
    data = stock_frame()
    data.loc[date, "DGS10"] = np.nan
    with pytest.raises(DataError, match="非treasury_closed"):
        evaluate(data, "HIKING", .8)


def test_early_close_publication_is_used_normally():
    data = stock_frame(end="2026-11-27")
    result = evaluate(data, "HIKING", .8)
    assert result["yield_calendar"]["yield_asof"] == "2026-11-27"
    assert result["yield_calendar"]["dgs10_status"] == "published"


def test_dgs10_more_than_five_nyse_sessions_stale_errors():
    data = stock_frame()
    data.loc[data.index[-6:], "DGS10"] = np.nan
    with pytest.raises(DataError, match="6 NYSE交易日 > 5"):
        evaluate(data, "HIKING", .8)


@pytest.mark.parametrize("yield_true", [True, False])
def test_move_unknown_leg_is_null_regardless_of_known_yield_leg(yield_true):
    data = stock_frame()
    if not yield_true:
        dates = data.DGS10.dropna().index[-20:]
        data.loc[dates, "DGS10"] = 4 + .5 * (-1.) ** np.arange(20)
    # MOVE只有最后3次公布；不参加DGS10完整性检查。
    out = yield_factors(data.index, data.DGS10, data.MOVE.tail(3))
    assert pd.isna(out.yield_stable.iloc[-1])
    assert pd.isna(out.move_yield_leg.iloc[-1])
    assert bool(out.yield_leg.iloc[-1]) is yield_true
    result = evaluate(data, "HIKING", .8, move_publications=data.MOVE.tail(3))
    assert result["yield_stable"] is None
    assert result["action"] != "EARNINGS_SLEEVE"
    assert decide(DecisionInputs("HIKING", "NONE", "NONE", .8, .25, None))["action"] == "DIP_ADD"


def test_future_dgs10_append_does_not_change_any_past_factor():
    data = stock_frame()
    past = yield_factors(data.index, data.DGS10, data.MOVE)
    future = pd.concat([data.DGS10, pd.Series([1000., -1000.],
                       index=pd.to_datetime(["2026-12-01", "2026-12-02"]))])
    changed = yield_factors(data.index, future, data.MOVE)
    pd.testing.assert_frame_equal(past, changed)
    a = evaluate(data, "HIKING", .8)
    b = evaluate(data, "HIKING", .8, dgs10_publications=future)
    assert a == b


def test_native_non_nyse_publication_is_not_lost_when_mapping():
    data = stock_frame(end="2026-04-06")
    native = data.DGS10.dropna()
    # 输入可包含NYSE不开市的公布日；原频率计算后才映射。
    native.loc[pd.Timestamp("2026-04-03")] = 5
    native = native.sort_index()
    out = yield_factors(data.index, native, data.MOVE)
    expected = native.diff().rolling(10).std().iloc[-1]
    assert out.yield_std10.iloc[-1] == pytest.approx(expected)


def test_holding_period_still_counts_nyse_sessions_on_treasury_holiday():
    assert trading_days_since(pd.Timestamp("2026-10-09").date(),
                              pd.Timestamp("2026-10-12").date()) == 1
    assert trading_days_since(pd.Timestamp("2026-10-09").date(),
                              pd.Timestamp("2026-10-23").date()) == 10
