"""Synthetic boundary/PIT checks; these do not validate investment performance."""
from copy import deepcopy
import json
import math

import exchange_calendars as xcals
import pandas as pd
import pytest

from macro_engine.surge import BUCKET_FIELDS, MIN_SAMPLE, TARGET, estimate, wilson


CAL = xcals.get_calendar("XNYS", start="2023-01-01", end="2025-01-01")
BUCKET = {"regime": "HIKING", "stress": "NONE", "dip_unit": .25,
          "yield_stable": True, "at_252_high": False}
FIELDS = {"symbol", "asof", "target", "p", "n", "ci_low", "ci_high", "method",
          "calibrated", "beats_baseline", "abstain", "reason"}


def sample(symbol, on="2024-04-01", following_close=106, **state):
    next_day = CAL.next_session(on).date().isoformat()
    row = {"symbol": symbol, "date": on, **BUCKET, **state,
           "available_at": CAL.session_close(on).isoformat(),
           "source": "synthetic-feature", "verified": True}
    prices = [{"date": day, "adj_close": close, "closed": True,
               "available_at": CAL.session_close(day).isoformat(),
               "source": "synthetic-adjusted-close", "verified": True}
              for day, close in [(on, 100), (next_day, following_close)]]
    return row, prices


def panel(n=30, *, winners=15, on="2024-04-01"):
    rows, prices = [], {}
    for i in range(n):
        symbol = f"S{i:03d}"
        row, series = sample(symbol, on, 106 if i < winners else 101)
        rows.append(row)
        prices[symbol] = series
    return {"pit_verified": True, "rows": rows, "prices": prices}


def run(data=None, *, asof="2024-04-03", bucket=None, symbol="NEW", **kwargs):
    return estimate(symbol, asof, BUCKET if bucket is None else bucket,
                    panel=data, **kwargs)


def test_schema_target_and_cross_symbol_pooled_frequency():
    result = run(panel())
    assert set(result) == FIELDS
    assert result["symbol"] == "NEW"  # The target need not be a historical sample.
    assert result["target"] == TARGET
    assert TARGET["price"] == "adj_close"
    assert TARGET["horizon_sessions"] == 1
    assert TARGET["return_threshold"] == .05
    assert result["n"] == 30 and result["p"] == .5
    assert result["ci_low"] == pytest.approx(.3315412564053376)
    assert result["ci_high"] == pytest.approx(.6684587435946624)
    assert result["calibrated"] is False
    assert result["beats_baseline"] is None
    assert result["abstain"] is False
    assert result["reason"] == "historical_frequency_uncalibrated"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("n", [0, 1, 29])
def test_insufficient_samples_return_null_probability(n):
    result = run(panel(n))
    assert result["n"] == n and result["p"] is None
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["abstain"] is True and result["reason"] == "insufficient_sample"


@pytest.mark.parametrize("data", [None, {}, {"pit_verified": False},
                                  {"pit_verified": 1, "rows": [], "prices": {}},
                                  {"pit_verified": True, "rows": [], "prices": []}])
def test_no_verified_pit_panel_abstains(data):
    result = run(data)
    assert result["reason"] == "no_pit_panel" and result["p"] is None
    assert result["n"] == 0 and result["abstain"] is True
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("symbol,kwargs", [(s, {}) for s in ["GLD", "IAU", "SLV", "SILJ", "TLT"]]
                         + [("600000", {"market": "CN"}), ("BND", {"asset_type": "BOND"}),
                            ("XAU", {"asset_type": "GOLD"})])
def test_excluded_target_never_computes_frequency(symbol, kwargs):
    result = run(panel(), symbol=symbol, **kwargs)
    assert result["reason"] == "excluded_asset" and result["n"] == 0
    assert result["p"] is None and result["abstain"] is True


def test_excluded_historical_assets_do_not_enter_pooled_sample():
    data = panel()
    for symbol, extra in [(s, {}) for s in ["GLD", "IAU", "SLV", "SILJ", "TLT"]] + [
        ("CN1", {"market": "CN"}), ("BND", {"asset_type": "BOND"})
    ]:
        row, prices = sample(symbol)
        row.update(extra)
        data["rows"].append(row)
        data["prices"][symbol] = prices
    assert run(data) == run(panel())


@pytest.mark.parametrize("field,value", [("regime", "FIRST_CUT"), ("stress", "TRIGGER"),
                                        ("dip_unit", .5), ("yield_stable", False),
                                        ("at_252_high", True)])
def test_only_exact_five_state_bucket_matches(field, value):
    data = panel(31, winners=15)
    data["rows"][-1][field] = value
    assert run(data) == run(panel())
    assert tuple(BUCKET_FIELDS) == tuple(BUCKET)


def test_extra_fields_never_become_a_sixth_bucket_or_input_probability():
    data = panel()
    for row in data["rows"]:
        row.update(p=.999, Y=1, label=1, label_available_at="1900-01-01T00:00:00Z",
                   quality_score=100, coefficient=500)
    wanted = {**BUCKET, "symbol": "NO_MATCH", "p": .01, "score": -999}
    assert run(data, bucket=wanted) == run(panel())


def test_exact_five_percent_close_is_success_and_high_is_not_a_label():
    data = panel()
    for i, prices in enumerate(data["prices"].values()):
        prices[-1]["adj_close"] = 105 if i < 15 else "104.9999999999999999999999999999"
        prices[-1]["high"] = 10000
        prices[-1]["close"] = 10000  # Only the adjusted-close series is used.
    assert run(data)["p"] == .5


def test_label_must_be_mature_by_previous_xnys_session_close():
    # Apr 1 feature + Apr 2 close label is unavailable to Apr 2 estimation.
    result = run(panel(), asof="2024-04-02")
    assert result["n"] == 0 and result["p"] is None
    assert run(panel(), asof="2024-04-03")["n"] == 30


def test_next_session_is_exchange_session_across_holiday_and_early_close():
    data = panel(on="2024-07-03")
    assert data["prices"]["S000"][0]["available_at"] == "2024-07-03T17:00:00+00:00"
    assert data["prices"]["S000"][1]["date"] == "2024-07-05"
    assert run(data, asof="2024-07-08")["n"] == 30
    assert run(data, asof="2024-07-05")["n"] == 0


def test_missing_next_session_does_not_skip_to_a_later_price():
    data = panel()
    prices = data["prices"]["S000"]
    prices[-1] = {**prices[-1], "date": "2024-04-03",
                  "available_at": CAL.session_close("2024-04-03").isoformat()}
    result = run(data, asof="2024-04-04")
    assert result["n"] == 29 and result["p"] is None


@pytest.mark.parametrize("leg", [0, 1])
@pytest.mark.parametrize("change", [{"available_at": "2024-04-03T20:01:00Z"},
                                     {"available_at": "2024-04-01T20:00:00"},
                                     {"closed": False}, {"verified": False},
                                     {"source": ""}, {"adj_close": 0},
                                     {"adj_close": float("nan")}])
def test_invalid_price_record_excluded_for_either_label_leg(leg, change):
    data = panel()
    data["prices"]["S000"][leg].update(change)
    result = run(data)
    assert result["n"] == 29 and result["p"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [{"available_at": "2024-04-01T20:00:01Z"},
                                     {"available_at": "2024-04-01T20:00:00"},
                                     {"date": "2024-03-29"},  # Good Friday
                                     {"verified": False}, {"source": ""},
                                     {"yield_stable": None}, {"at_252_high": 0},
                                     {"dip_unit": float("nan")}])
def test_invalid_or_not_originally_known_feature_excluded(change):
    data = panel()
    data["rows"][0].update(change)
    assert run(data)["n"] == 29


def test_future_append_and_backfilled_revisions_cannot_change_past_result():
    data = panel()
    before = run(data)
    future_stamp = "2024-04-04T20:00:00Z"
    for row in list(data["rows"]):
        data["rows"].append({**row, "dip_unit": 0, "available_at": future_stamp})
    for prices in data["prices"].values():
        prices.extend([{**p, "adj_close": 1000000, "available_at": future_stamp}
                       for p in list(prices)])
        prices.append({**prices[0], "date": "2024-04-04", "available_at": future_stamp})
    row, prices = sample("FUTURE", "2024-04-03")
    data["rows"].append(row)
    data["prices"]["FUTURE"] = prices
    # Even an absurdly old backfill must be filtered before calendar creation.
    data["rows"].append({**data["rows"][0], "date": "0001-01-01",
                         "available_at": future_stamp})
    assert run(data) == before


def test_available_revision_after_own_session_close_is_also_excluded():
    data = panel()
    before = run(data, asof="2024-04-04")
    data["rows"].append({**data["rows"][0], "stress": "TRIGGER",
                         "available_at": "2024-04-02T20:00:00Z"})
    data["prices"]["S000"].append({**data["prices"]["S000"][0], "adj_close": 1,
                                   "available_at": "2024-04-02T20:00:00Z"})
    assert run(data, asof="2024-04-04") == before


def test_duplicates_are_idempotent_and_equal_timestamp_conflicts_abstain():
    data = panel()
    before = run(data)
    data["rows"].append(deepcopy(data["rows"][0]))
    data["prices"]["S000"].append(deepcopy(data["prices"]["S000"][0]))
    assert run(data) == before
    data["rows"].append({**data["rows"][0], "stress": "TRIGGER"})
    assert run(data)["n"] == 29
    data = panel()
    data["prices"]["S000"].append({**data["prices"]["S000"][0], "adj_close": 10})
    assert run(data)["n"] == 29


def test_latest_intraday_feature_version_determines_bucket():
    data = panel()
    data["rows"].append({**data["rows"][0], "stress": "TRIGGER",
                         "available_at": "2024-04-01T19:59:00Z"})
    assert run(data)["n"] == 30  # The close-time original remains the latest.
    data["rows"][0]["available_at"] = "2024-04-01T19:58:00Z"
    assert run(data)["n"] == 29


@pytest.mark.parametrize("asof", ["2024-03-29", "2024-04-06", "bad-date",
                                 "2024-04-03T12:00:00Z"])
def test_asof_must_be_real_xnys_session_date(asof):
    result = run(panel(), asof=asof)
    assert result["reason"] == "invalid_asof" and result["p"] is None


@pytest.mark.parametrize("bucket", [{}, {**BUCKET, "yield_stable": None},
                                    {**BUCKET, "dip_unit": float("inf")},
                                    {**BUCKET, "dip_unit": .123},
                                    {**BUCKET, "regime": "continuous_score_0.7"},
                                    {**BUCKET, "stress": "continuous_score_0.9"}])
def test_missing_or_nonfinite_bucket_abstains_without_a_score(bucket):
    result = run(panel(), bucket=bucket)
    assert result["reason"] == "invalid_bucket" and result["p"] is None


def test_invalid_discrete_state_rows_cannot_enter_training():
    data = panel()
    data["rows"][0]["dip_unit"] = .123
    result = run(data)
    assert result["n"] == 29
    assert result["p"] is None and result["abstain"] is True


@pytest.mark.parametrize("symbol", ["GOLD", "SILVER"])
def test_metals_never_get_a_stock_frequency(symbol):
    result = run(panel(), symbol=symbol)
    assert result["p"] is None and result["abstain"] is True
    assert result["reason"] == "excluded_asset"


def test_wilson_boundary_values_and_validation():
    assert MIN_SAMPLE == 30
    assert wilson(0, 0) == (None, None)
    low, high = wilson(0, 30)
    assert low == pytest.approx(0, abs=1e-15)
    assert high == pytest.approx(.1135133931739688)
    low, high = wilson(30, 30)
    assert low == pytest.approx(.8864866068260312) and high == pytest.approx(1)
    assert all(math.isfinite(x) for x in wilson(1, 30))
    for args in [(31, 30), (-1, 30), (1, -1), (True, 30), (1.5, 30)]:
        with pytest.raises(ValueError):
            wilson(*args)
