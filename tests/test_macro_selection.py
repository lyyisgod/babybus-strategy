"""合成表驱动功能验证，非投资回测。"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_selection.selection import select, gate, fundamental
from macro_selection.policy import official_regime
from macro_selection.bridge import lagged_oas, instrument_rules
from macro_selection.prepare import price_facts
from macro_selection.snapshot import latest_closed_nyse, run


DATE = "2026-10-02"
START = "2026-07-07"
CAL = {"US": {"last_close": DATE, "open_for_target": True, "source": "XNYS"},
       "CN": {"last_close": "2026-09-30", "open_for_target": False,
              "next_open": "2026-10-08", "source": "SSE official"}}


def macro(**overrides):
    m = {"asof": DATE, "version": "v1.1", "data_valid": True, "regime": "HIKING",
         "regime_source": "official", "stress": {"state": "NONE"}, "divergence": "NONE",
         "yield_stable": False, "constraints": {"allow_margin": True, "max_gross": None},
         "asset_rules": {"AAA": {"size_unit": .25, "asof": DATE}},
         "factors": {"dgs10_up_5d": True, "dgs10": 4.}}
    m.update(overrides)
    return m


def candidate(symbol="AAA", *, market="US", role="核心", template="growth", **overrides):
    date = DATE if market == "US" else "2026-09-30"
    facts = {"ret_1d": -.02, "ret_5d": -.04, "drawdown_63d": .20,
             "below_ma50": True, "below_ma200": False, "rsi14": 25,
             "new_low20": True, "hike_beta": -.2, "liquid": True,
             "dividend_yield": .06, "mature_company": True,
             "roe": .12, "revenue_growth": .05, "high_valuation": True,
             "financing_sensitive": True}
    c = {"symbol": symbol, "market": market, "role": role, "asof": date,
         "window_start": START, "template": template, "in_smh": role == "核心",
         "smh_membership": {"member": role == "核心", "date": date, "source": "official holdings fixture"},
         "membership": {"universe": "SMH" if market == "US" else "CSI300", "date": date, "source": "official holdings fixture"},
         "consensus": {"baseline_eps": 10., "current_eps": 9.9,
                       "baseline_date": START, "date": date, "fiscal_period": "FY2027",
                       "baseline_fiscal_period": "FY2027", "source": "consensus fixture",
                       "series": "consensus.FY2027.EPS", "is_consensus": True,
                       "baseline_available_at": START + ("T15:00:00-04:00" if market == "US" else "T14:00:00+08:00"),
                       "available_at": date + ("T15:00:00-04:00" if market == "US" else "T14:00:00+08:00")},
         "facts": {k: {"value": v, "date": date, "series": symbol + "." + k,
                        "source": "fixture", "closed": True} for k, v in facts.items()}}
    c.update(overrides)
    return c


@pytest.mark.parametrize("regime,stress,div,valid,expected", [
    ("HIKING", "NONE", "NONE", True, (True, True, False)),
    ("HIKING", "WATCH", "NONE", True, (True, True, False)),
    ("HIKING", "PENDING", "NONE", True, (True, True, False)),
    ("PAUSE", "NONE", "NONE", True, (True, True, False)),
    ("FIRST_CUT", "NONE", "NONE", True, (False, False, False)),
    ("EASING", "NONE", "NONE", True, (False, False, False)),
    ("UNKNOWN", "NONE", "NONE", True, (False, False, False)),
    ("HIKING", "TRIGGER", "NONE", True, (False, False, True)),
    ("HIKING", "NONE", "CONFIRMED_SELLOFF", True, (False, False, True)),
    ("HIKING", "NONE", "NONE", False, (False, False, False)),
])
def test_macro_gate_exact(regime, stress, div, valid, expected):
    g = gate(macro(regime=regime, stress={"state": stress}, divergence=div, data_valid=valid))
    assert tuple(g[k] for k in ("allow_new_core", "allow_new_satellite", "existing_core_only")) == expected
    if stress == "TRIGGER" or div == "CONFIRMED_SELLOFF":
        assert g["allow_margin"] is False and g["max_gross"] == 1


@pytest.mark.parametrize("ret,gross,existing,action,status,size", [
    (0., .8, True, "HOLD", "等阴线", 0),
    (-.02, .8, True, "CORE_ADD_NO_MARGIN", "可执行", .25),
    (-.02, 1.4, True, "DELEVER_TO_1X", "删除", 0),
    (-.02, None, True, "HOLD", "观察", 0),
    (-.02, .8, False, "HOLD", "删除", 0),
])
def test_pressure_actions_and_sizing(ret, gross, existing, action, status, size):
    c = candidate(existing_core=existing)
    c["facts"]["ret_1d"]["value"] = ret
    out = select(macro(stress={"state": "TRIGGER"}), [c], calendars=CAL, gross_exposure=gross)
    row = out["rows"][0]
    assert (row["action"], row["status"], row["size_unit"]) == (action, status, size)
    assert out["gate"]["allow_margin"] is False


@pytest.mark.parametrize("revision,expected", [(0, True), (-.05, True), (-.051, False), (.10, True)])
def test_mispricing_formula_not_probability(revision, expected):
    c = candidate(); c["consensus"]["current_eps"] = 10 * (1 + revision)
    good, score, actual, _ = fundamental(c)
    assert good is expected
    assert score == pytest.approx(.20 - max(0, -revision))
    assert actual == pytest.approx(revision)


@pytest.mark.parametrize("problem", ["missing", "different_fy", "future", "after_close", "old_endpoint", "not_consensus"])
def test_news_cannot_replace_consensus(problem):
    c = candidate()
    if problem == "missing": c["consensus"] = {}
    if problem == "different_fy": c["consensus"]["baseline_fiscal_period"] = "FY2026"
    if problem == "future": c["consensus"]["available_at"] = "2026-10-05T14:00:00-04:00"
    if problem == "after_close": c["consensus"]["available_at"] = "2026-10-02T16:01:00-04:00"
    if problem == "old_endpoint": c["consensus"]["baseline_date"] = "2026-07-06"
    if problem == "not_consensus": c["consensus"]["is_consensus"] = False
    c["news"] = "extremely positive story"
    out = select(macro(), [c], calendars=CAL)
    assert not out["core"] and out["rows"][0]["fundamental_gate"] is None


def test_rank_top_one_and_no_story_tiebreak():
    a, b = candidate(), candidate("BBB")
    b["facts"]["drawdown_63d"]["value"] = .25
    assert select(macro(), [a, b], calendars=CAL)["core"] == {"US": "BBB"}
    b["facts"]["drawdown_63d"]["value"] = .20
    b["story_score"] = 999
    assert select(macro(), [a, b], calendars=CAL)["core"] == {}
    for c, score in ((a, 1), (b, 2)):
        c["facts"]["secondary_score"] = {"value": score, "date": DATE, "source": "fixture", "series": "quality", "closed": True}
    assert select(macro(), [a, b], calendars=CAL)["core"] == {"US": "BBB"}


def test_positive_core_waits_and_chase_is_removed():
    c = candidate(); c["facts"]["ret_1d"]["value"] = .01
    out = select(macro(), [c], calendars=CAL)
    assert out["core"] == {"US": "AAA"}
    assert out["rows"][0]["status"] == "等阴线" and out["rows"][0]["size_unit"] == 0
    c["facts"]["ret_5d"]["value"] = .10
    out = select(macro(), [c], calendars=CAL)
    assert out["core"] == {} and out["rows"][0]["status"] == "删除"


def test_yield_null_not_earnings_and_yield_true_disables_satellites():
    assert gate(macro(yield_stable=None))["allow_new_satellite"] is True
    assert gate(macro(yield_stable=True))["allow_new_satellite"] is False
    c = candidate(); c["consensus"]["current_eps"] = 10.1
    assert select(macro(yield_stable=True), [c], calendars=CAL)["core"] == {"US": "AAA"}
    c["consensus"]["current_eps"] = 9.9
    assert select(macro(yield_stable=True), [c], calendars=CAL)["core"] == {"US": "AAA"}
    c["consensus"] = {}
    assert select(macro(yield_stable=True), [c], calendars=CAL)["core"] == {}


def test_dividend_and_growth_scores_separate_and_total_cap():
    cs = [candidate(s, role="卫星", membership={}) for s in ("AAA", "BBB", "CCC", "DDD")]
    cs.append(candidate("EEE", role="卫星", membership={}, template="dividend", consensus={}))
    out = select(macro(), cs, calendars=CAL)
    assert len(out["satellites"]) == 3
    row = next(r for r in out["rows"] if r["symbol"] == "EEE")
    assert row["dividend_premium"] == pytest.approx(.02) and row["mispricing_score"] is None
    assert row["suggested_nominal_cap"] == .02 and row["size_unit"] == 0


def test_unknown_and_invalid_macro_do_not_consume_candidates():
    class Forbidden:
        def __iter__(self): raise AssertionError("must stop before stocks")
    for m in (macro(regime="UNKNOWN"), macro(data_valid=False), macro(stress={"state": "PENDING"}, critical_null=True)):
        assert select(m, Forbidden(), calendars=CAL)["rows"] == []


def test_china_holiday_observation_only_and_penalty():
    c = candidate(market="CN"); c["consensus"]["current_eps"] = 10.
    out = select(macro(), [c], calendars=CAL)
    row = out["rows"][0]
    assert out["core"] == {"CN": "AAA"}
    assert row["mispricing_score"] == pytest.approx(.15)
    assert row["status"] == "观察" and row["size_unit"] == 0 and row["action"] == "HOLD"
    c["facts"]["roe"]["value"] = None
    assert select(macro(), [c], calendars=CAL)["core"] == {}


def statements(older="HOLD", latest="HIKE"):
    previous = [3.5, 3.75]
    current = {"HIKE": [3.75, 4.], "HOLD": previous, "CUT": [3.25, 3.5]}[latest]
    return [{"date": "2026-07-29", "action": older, "target_range_pct": previous,
             "source": "https://www.federalreserve.gov/previous", "original_text": "fixture", "verified": True},
            {"date": "2026-09-16", "action": latest, "target_range_pct": current,
             "source": "https://www.federalreserve.gov/latest", "original_text": "fixture", "verified": True}]


def probabilities(cut=0.):
    return {"source_name": "CME FedWatch", "source": "https://www.cmegroup.com/fedwatch",
            "verified": True, "meeting_date": "2026-10-28", "market_date": DATE,
            "data_asof": "2026-10-03T01:42:40-05:00",
            "probabilities": {"cut": cut, "hold": 1-cut, "hike": 0.}}


@pytest.mark.parametrize("older,latest,cut,expected", [
    ("HOLD", "HIKE", .8, "HIKING"), ("HIKE", "HOLD", .49, "PAUSE"),
    ("HIKE", "HOLD", .5, "UNKNOWN"), ("HIKE", "CUT", 0, "FIRST_CUT"),
    ("CUT", "CUT", 0, "EASING"), ("HOLD", "CUT", 0, "UNKNOWN"),
])
def test_official_policy(older, latest, cut, expected):
    assert official_regime(statements(older, latest), probabilities(cut), DATE)["regime"] == expected


def test_missing_conflicting_or_stale_policy_never_filled_hiking():
    assert official_regime(statements(), None, DATE)["regime"] == "UNKNOWN"
    s = statements(); s[-1]["target_range_pct"] = [3.5, 3.75]
    assert official_regime(s, probabilities(), DATE)["regime"] == "UNKNOWN"
    p = probabilities(); p["market_date"] = "2026-10-01"
    assert official_regime(statements(), p, DATE)["regime"] == "UNKNOWN"


def test_oas_lag_publication_day_and_future_append():
    dates = pd.bdate_range("2026-09-01", periods=12)
    pubs = pd.Series(np.arange(12, dtype=float), index=dates)
    out = lagged_oas(pubs, dates)
    assert pd.isna(out.oas_level.iloc[0])
    assert out.oas_level.iloc[-1] == 10 and out.oas_delta5.iloc[-1] == 5
    assert out.oas_used_publication.iloc[-1] == dates[-2]
    future = pd.concat([pubs, pd.Series([999.], index=[dates[-1] + pd.Timedelta(days=10)])])
    pd.testing.assert_frame_equal(lagged_oas(future, dates), out)
    missing_day = dates[7]
    holidays = pubs.drop(missing_day)
    mapped = lagged_oas(holidays, dates)
    assert mapped.oas_used_publication.loc[missing_day] == dates[5]


def test_prepare_uses_closed_cutoff_and_same_63_day_eps_anchor():
    dates = pd.bdate_range(end=DATE, periods=210)
    prices = pd.Series(np.linspace(100, 80, len(dates)), index=dates)
    out = price_facts("AAA", prices, DATE, "fixture")
    assert out["window_start"] == dates[-63].date().isoformat()
    assert out["facts"]["drawdown_63d"]["value"] == pytest.approx(1 - 80/prices.iloc[-63])
    future = pd.concat([prices, pd.Series([1000.], index=[pd.Timestamp("2026-10-05")])])
    assert price_facts("AAA", future, DATE, "fixture") == out


def test_runtime_uses_last_closed_session_not_intraday_or_weekend():
    assert latest_closed_nyse("2026-10-03T19:00:00Z").date().isoformat() == DATE
    assert latest_closed_nyse("2026-10-02T18:00:00Z").date().isoformat() == "2026-10-01"
    assert latest_closed_nyse("2026-10-02T20:01:00Z").date().isoformat() == DATE


def test_live_failure_stops_before_candidate_loader(monkeypatch, tmp_path):
    import macro_selection.snapshot as entry
    def missing(*a, **kw): raise ValueError("DGS10 missing")
    def forbidden(): raise AssertionError("cannot scan stocks")
    monkeypatch.setattr(entry, "fetch_dataset", missing)
    out = run(datetime(2026, 10, 3, 19, tzinfo=timezone.utc),
              {"statements": statements(), "next_meeting_probs": probabilities()}, CAL,
              tmp_path, candidate_loader=forbidden)
    assert out["macro"]["regime"] == "HIKING"
    assert out["macro"]["data_valid"] is False and out["rows"] == []


def test_all_macro_v11_files_unchanged():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "docs/frozen_v1_1.sha256"
    for line in manifest.read_text().splitlines():
        expected, path = line.split(maxsplit=1)
        assert hashlib.sha256((root / path.strip()).read_bytes()).hexdigest() == expected


def test_selector_does_not_import_or_recompute_macro():
    import macro_selection.selection as module
    text = Path(module.__file__).read_text()
    assert "from macro_regime" not in text and "import macro_regime" not in text
    assert "pressure_combinations(" not in text and "dip_units(" not in text
    assert instrument_rules({"AAA": {"ret_1d": -.02, "ret_5d": -.04, "asof": DATE}})["AAA"]["size_unit"] == .25


def test_bridge_pending_trigger_and_two_day_release():
    from types import SimpleNamespace
    from macro_selection.bridge import macro_output
    from test_macro_regime_yield_calendar import stock_frame
    f = stock_frame(end="2026-10-06")
    f["IXG"] = 150.; f["MOVE"] = 100.
    f.loc["2026-10-01":"2026-10-02", "IXG"] = 124.
    f.loc["2026-10-01":"2026-10-02", "MOVE"] = 130.
    for date, expected in (("2026-10-01", "PENDING"), ("2026-10-02", "TRIGGER"),
                           ("2026-10-05", "TRIGGER"), ("2026-10-06", "NONE")):
        frame = f.loc[:date]
        dataset = SimpleNamespace(frame=frame, metadata={"series": {}},
                                  dgs10_publications=f.DGS10, move_publications=f.MOVE,
                                  dff_publications=f.DFF)
        oas = pd.Series(np.linspace(3, 4, len(f)), index=f.index)
        out = macro_output(dataset, {"regime": "HIKING", "regime_source": "official"},
                           oas_publications=oas, oas_metadata={"publication_dates_verified": True})
        assert out["stress"]["state"] == expected
        assert out["constraints"]["allow_margin"] is (expected != "TRIGGER")
        assert out["stress"]["author_combo"] is (date <= "2026-10-02")


def test_unverified_oas_schedule_rejected():
    from macro_selection.bridge import macro_output
    with pytest.raises(ValueError, match="publication schedule unverified"):
        macro_output(None, {}, oas_publications=None, oas_metadata={})
