"""Account-rule contracts using raw synthetic asset evidence.

Only the macro adapter is replaced with already-valid macro facts, isolating
daily account rules.  The real engine, policy classifier, evidence readers,
price calculations and immutable Decision are exercised throughout.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import exchange_calendars as xcals
import pandas as pd
import pytest

import macro_engine.engine as engine


ASOF = "2026-10-02"
CAL = xcals.get_calendar("XNYS", start="2024-01-01", end="2027-01-01")
DAYS = list(CAL.sessions_in_range(CAL.first_session, ASOF)[-300:].strftime("%Y-%m-%d"))
WINDOW_START = DAYS[-63]
POLICY = [{"date": "2026-01-28", "action": "HIKE", "target_range": [4.5, 4.75],
           "source": "https://www.federalreserve.gov/newsevents/pressreleases/synthetic.htm",
           "verified": True}]


def record(day, **values):
    return {"date": day, "available_at": CAL.session_close(day).isoformat(),
            "source": "synthetic-PIT", "verified": True, **values}


def price_rows(last=98, previous=100, *, days=DAYS, market="US"):
    values = [100.] * len(days)
    values[-2:] = [previous, last]
    if market == "US":
        return [record(day, adj_open=100., adj_close=value, closed=True) for day, value in zip(days, values)]
    return [{"date": day, "adj_open": 100., "adj_close": value, "closed": True, "verified": True,
             "source": "synthetic-CN-PIT", "available_at": day + "T15:00:00+08:00"}
            for day, value in zip(days, values)]


def asset(symbol="AAA", role="CORE", industry="SEMICONDUCTOR", nominal=20000,
          market="US", **extra):
    return {"symbol": symbol, "market": market, "role": role, "industry": industry,
            "nominal": nominal, "opened_on": DAYS[0], **extra}


def base(*, core=True, last=98, previous=100):
    book = {"nav": 100000, "budget": 10000,
            "positions": [asset()] if core else [], "watchlist": []}
    facts = {"prices": {"AAA": price_rows(last, previous)},
             "memberships": {"SMH": [record(DAYS[0], symbol="AAA", member=True)]},
             "consensus": {"AAA": [
                 record(WINDOW_START, eps=10., fiscal_period="FY2027", is_consensus=True),
                 record(ASOF, eps=9.8, fiscal_period="FY2027", is_consensus=True),
             ]}}
    return book, facts


@pytest.fixture
def macro(monkeypatch):
    value = {"stress": {"state": "NONE", "trigger_on": False}, "pressure": False,
             "yield_stable": False, "data_valid": True,
             "trace": {"divergence": "NONE", "synthetic": True}}
    monkeypatch.setattr(engine, "macro_facts", lambda facts, asof: deepcopy(value))
    return value


def decide(book, facts, *, asof=ASOF):
    return engine.decide(book, facts, POLICY, asof)


def add_satellite(book, facts, symbol="TLT", *, held=False, industry="TREASURY", last=95):
    (book["positions"] if held else book["watchlist"]).append(
        asset(symbol, "SATELLITE", industry, 1000 if held else 0))
    facts["prices"][symbol] = price_rows(last)


def rejection(decision, symbol, reason):
    return any(r["symbol"] == symbol and r["reason"] == reason for r in decision.rejects)


def test_ledger_core_is_not_replaced_by_a_stock_with_a_larger_drop(macro):
    book, facts = base()
    book["watchlist"].append(asset("BBB", "SATELLITE", nominal=0))
    facts["prices"]["BBB"] = price_rows(80)
    facts["memberships"]["SMH"].append(record(DAYS[0], symbol="BBB", member=True))
    result = decide(book, facts)
    assert result.action == "CORE_ADD" and result.size_unit == .25
    assert [row["symbol"] for row in result.cores] == ["AAA"]
    assert book["positions"][0]["role"] == "CORE"
    assert book["watchlist"][0]["role"] == "SATELLITE"


def test_multiple_ledger_cores_never_create_a_best_dip_override(macro):
    book, facts = base()
    book["watchlist"].append(asset("BBB", "CORE", nominal=0))
    facts["prices"]["BBB"] = price_rows(80)
    facts["memberships"]["SMH"].append(record(DAYS[0], symbol="BBB", member=True))
    result = decide(book, facts)
    assert result.action == "OBSERVE" and result.size_unit == 0
    assert not result.cores
    assert "multiple_global_cores" in result.trace["book_reasons"]


def test_one_account_action_key_even_with_core_and_satellite_opportunities(macro):
    book, facts = base()
    add_satellite(book, facts)
    result = decide(book, facts)
    data = json.loads(result.to_json())
    assert list(data).count("action") == 1
    assert data["action"] == "CORE_ADD"
    assert all("action" not in row for key in ("cores", "satellites", "rejects") for row in data[key])
    assert all(row["status"] == "观察" for row in result.satellites)
    assert decide(book, facts).to_dict() == result.to_dict()


@pytest.mark.parametrize("gross", [.2, 1.])
@pytest.mark.parametrize("divergence", ["NONE", "CONFIRMED_SELLOFF"])
def test_pressure_only_changes_margin_and_gross_constraints_below_one(macro, gross, divergence):
    book, facts = base()
    book["positions"][0]["nominal"] = gross * book["nav"]
    clear = decide(book, facts)
    macro.update(pressure=True, stress={"state": "TRIGGER", "trigger_on": True},
                 trace={"divergence": divergence})
    result = decide(book, facts)
    assert clear.action == "CORE_ADD" and result.action == "CORE_ADD_NO_MARGIN"
    assert clear.size_unit == result.size_unit == .25
    assert result.constraints["allow_margin"] is False
    assert result.constraints["max_gross"] == 1.
    assert result.cores[0]["status"] == "可执行"


def test_pressure_above_one_produces_only_deleveraging(macro):
    book, facts = base()
    book["positions"][0]["nominal"] = 110000
    add_satellite(book, facts)
    macro.update(pressure=True, stress={"state": "TRIGGER", "trigger_on": True},
                 trace={"divergence": "CONFIRMED_SELLOFF"})
    result = decide(book, facts)
    assert result.action == "DELEVER_TO_1X" and result.size_unit == 0
    assert not result.cores and not result.satellites
    assert result.constraints["allow_margin"] is False
    assert result.constraints["max_gross"] == 1.


def test_caller_returns_and_wrongly_sold_scores_cannot_override_raw_prices(macro):
    book, facts = base()
    before = decide(book, facts)
    facts.update(ret_1d=.99, ret_5d=.99, dip_unit=0, score=100, wrong_sell_score=100,
                 divergence="CONFIRMED_SELLOFF", surge_p=.99)
    for p in book["positions"]:
        p.update(ret_1d=-.5, ret_5d=-.9, dip_unit=1, wrong_sell_score=100)
    for row in facts["prices"]["AAA"]:
        row.update(ret_1d=.99, ret_5d=.99, close=100000, score=100)
    assert decide(book, facts).to_dict() == before.to_dict()
    assert before.trace["computed_price_facts"]["AAA"]["ret_1d"] == -.02


def test_core_previous_close_decline_but_positive_intraday_candle_cannot_add(macro):
    book, facts = base(last=98)
    facts["prices"]["AAA"][-1]["adj_open"] = 97
    result = decide(book, facts)
    assert result.trace["computed_price_facts"]["AAA"]["ret_1d"] == -.02
    assert result.trace["computed_price_facts"]["AAA"]["down_candle"] is False
    assert result.action == "HOLD" and result.size_unit == 0


def test_core_without_verified_adjusted_open_remains_observation(macro):
    book, facts = base()
    facts["prices"]["AAA"][-1].pop("adj_open")
    result = decide(book, facts)
    assert result.action == "HOLD" and result.size_unit == 0
    assert all(row["status"] != "可执行" for row in result.cores)


@pytest.mark.parametrize("daily,expected", [(-.01, 0), (-.0101, .25), (-.03, .25),
                                          (-.0301, .5), (-.06, .5), (-.0601, 1)])
def test_frozen_core_dip_tiers_from_adjusted_closes(macro, daily, expected):
    book, facts = base(last=100 * (1 + daily))
    result = decide(book, facts)
    assert result.size_unit == expected
    assert result.action == ("CORE_ADD" if expected else "HOLD")


@pytest.mark.parametrize("current,allowed", [(9.5, True), (9.5000000000001, True),
                                           (9.4999999999999, False), (9.499, False)])
def test_same_fiscal_consensus_downgrade_boundary_is_inclusive(macro, current, allowed):
    book, facts = base()
    facts["consensus"]["AAA"][-1]["eps"] = current
    result = decide(book, facts)
    assert (result.action == "CORE_ADD") is allowed
    if not allowed:
        assert rejection(result, "AAA", "eps_downgrade_gt_5pct")


def test_consensus_cannot_cross_fiscal_periods(macro):
    book, facts = base()
    facts["consensus"]["AAA"][-1]["fiscal_period"] = "FY2028"
    result = decide(book, facts)
    assert result.action == "HOLD"
    assert rejection(result, "AAA", "same_fiscal_consensus_baseline_missing")


def test_hold_after_first_cut_keeps_metals_on_their_own_clock(macro):
    book, facts = base()
    add_satellite(book, facts, "SLV", held=True, industry="PRECIOUS_METALS")
    statements = deepcopy(POLICY) + [
        {**POLICY[0], "date": "2026-09-16", "action": "CUT", "target_range": [4.25, 4.5]},
        {**POLICY[0], "date": "2026-09-30", "action": "HOLD", "target_range": [4.25, 4.5]},
    ]
    result = engine.decide(book, facts, statements, ASOF)
    assert result.regime == "EASING" and result.trace["policy"]["cut_index"] == 1
    assert result.trace["metal_clock"] == "HOLD_FIRST_CUT"
    assert result.action == "PRECIOUS_METALS_HOLD"
    assert result.size_unit == 0
    assert not result.satellites


def test_raw_stock_price_history_is_not_limited_to_three_years():
    from macro_engine.inputs import prices
    calendar = xcals.get_calendar("XNYS", start="2021-01-01", end="2027-01-01")
    days = calendar.sessions_in_range("2021-01-04", ASOF)
    rows = [{"date": day.date().isoformat(), "adj_close": 100., "adj_open": 101.,
             "closed": True, "verified": True, "source": "synthetic-long-history",
             "available_at": calendar.session_close(day).isoformat()} for day in days]
    series = prices({"prices": {"AAA": rows}}, "AAA", ASOF)
    assert len(series) == len(days)
    assert series.index[0].date().isoformat() == "2021-01-04"


def test_future_available_consensus_and_price_revisions_are_inert(macro):
    book, facts = base()
    before = decide(book, facts).to_dict()
    facts["consensus"]["AAA"].append({**facts["consensus"]["AAA"][-1], "eps": .01,
                                      "available_at": "2026-10-05T20:00:00Z"})
    facts["prices"]["AAA"].append({**facts["prices"]["AAA"][-1], "adj_close": .01,
                                   "available_at": "2026-10-05T20:00:00Z"})
    assert decide(book, facts).to_dict() == before


def test_consensus_baseline_must_already_be_available_at_window_start(macro):
    book, facts = base()
    facts["consensus"]["AAA"][0]["available_at"] = DAYS[-62] + "T20:00:00Z"
    result = decide(book, facts)
    assert result.action == "HOLD"
    assert rejection(result, "AAA", "same_fiscal_consensus_baseline_missing")


def test_membership_exit_starts_on_its_effective_date(macro):
    book, facts = base()
    before = decide(book, facts).to_dict()
    facts["memberships"]["SMH"].append(record(ASOF, symbol="AAA", member=False,
                                               date="2026-10-05"))
    assert decide(book, facts).to_dict() == before
    facts["memberships"]["SMH"][-1]["date"] = ASOF
    result = decide(book, facts)
    assert result.action == "HOLD"
    assert rejection(result, "AAA", "not_current_smh_member")


def test_future_available_membership_exit_is_ignored(macro):
    book, facts = base()
    before = decide(book, facts).to_dict()
    facts["memberships"]["SMH"].append(record(ASOF, symbol="AAA", member=False,
                                               available_at="2026-10-05T20:00:00Z"))
    assert decide(book, facts).to_dict() == before


@pytest.mark.parametrize("core", [False, True])
def test_sixth_industry_name_is_rejected_for_every_new_target(macro, core):
    book, facts = base(core=False)
    book["positions"] = [asset(f"OLD{i}", "SATELLITE", "SEMICONDUCTOR" if core else "UTILITY", 1000)
                         for i in range(5)]
    if core:
        book["watchlist"] = [asset(nominal=0)]
        symbol = "AAA"
    else:
        add_satellite(book, facts, "UTIL", industry="UTILITY")
        symbol = "UTIL"
    result = decide(book, facts)
    if core:
        row = next(row for row in result.cores if row["symbol"] == symbol)
        assert row["reason"] == "sixth_industry_name" and row["status"] == "观察"
    else:
        assert rejection(result, symbol, "sixth_industry_name")
    assert result.action not in {"CORE_ADD", "CORE_ADD_NO_MARGIN", "SATELLITE_DIP"}


@pytest.mark.parametrize("core", [False, True])
def test_total_sixteen_name_hard_cap_applies_to_new_core_and_satellite(macro, core):
    book, facts = base(core=False)
    book["positions"] = [asset(f"OLD{i}", "SATELLITE", f"LEGACY{i}", 1000) for i in range(16)]
    if core:
        book["watchlist"] = [asset(nominal=0)]
        symbol = "AAA"
    else:
        add_satellite(book, facts)
        symbol = "TLT"
    result = decide(book, facts)
    if core:
        row = next(row for row in result.cores if row["symbol"] == symbol)
        assert row["reason"] == "total_name_cap" and row["status"] == "观察"
    else:
        assert rejection(result, symbol, "total_name_cap")
    assert result.action not in {"CORE_ADD", "CORE_ADD_NO_MARGIN", "SATELLITE_DIP"}


def test_book_cannot_raise_hard_cap_above_sixteen(macro):
    book, facts = base()
    book["max_names"] = 17
    result = decide(book, facts)
    assert result.constraints["max_names"] == 16
    assert result.action == "OBSERVE"
    assert "invalid_total_name_cap" in result.trace["book_reasons"]


def test_existing_core_dip_does_not_create_a_sixth_or_seventeenth_name(macro):
    book, facts = base()
    book["positions"].extend(asset(f"OLD{i}", "SATELLITE",
                                    "SEMICONDUCTOR" if i < 4 else f"LEGACY{i}", 1000)
                             for i in range(15))
    result = decide(book, facts)
    assert result.action == "CORE_ADD" and result.size_unit == .25
    assert len(book["positions"]) == 16


@pytest.mark.parametrize("field", ["budget", "nav", "nominal"])
def test_missing_account_budget_or_nominal_is_not_executable(macro, field):
    book, facts = base()
    (book["positions"][0] if field == "nominal" else book).pop(field)
    result = decide(book, facts)
    assert result.action == "OBSERVE" and result.size_unit == 0
    assert result.trace["budget_ready"] is False
    assert result.constraints["allow_new_core"] is False
    assert all(row["status"] != "可执行" for row in result.cores + result.satellites)


@pytest.mark.parametrize("earnings,expected", [(None, "CORE_ADD"), ("2026-10-01", "CORE_ADD"),
                                             ("2026-10-02", "CORE_ADD"),
                                             ("2026-10-05", "HOLD")])
def test_future_earnings_hold_requires_an_actual_future_date(macro, earnings, expected):
    book, facts = base()
    if earnings is not None:
        facts["earnings"] = {"AAA": {"date": earnings}}
    result = decide(book, facts)
    assert result.action == expected
    assert (result.cores[0]["reason"] == "before_earnings_hold_only") is (expected == "HOLD")


def test_core_at_252_high_is_retained_without_satellite_high_rejection(macro):
    book, facts = base(last=101)
    result = decide(book, facts)
    assert result.action == "HOLD"
    assert result.cores[0]["metrics"]["at_252_high"] is True
    assert not rejection(result, "AAA", "new_satellite_at_252_high")


@pytest.mark.parametrize("held", [False, True])
def test_yield_stability_blocks_only_new_satellite_names(macro, held):
    book, facts = base(core=False)
    add_satellite(book, facts, held=held)
    macro["yield_stable"] = True
    result = decide(book, facts)
    if held:
        assert result.action == "SATELLITE_DIP"
        assert [row["symbol"] for row in result.satellites] == ["TLT"]
        assert not rejection(result, "TLT", "yield_stable_no_new_satellites")
    else:
        assert result.action == "HOLD"
        assert rejection(result, "TLT", "yield_stable_no_new_satellites")


def test_yield_stability_does_not_block_core_dip(macro):
    book, facts = base()
    macro["yield_stable"] = True
    result = decide(book, facts)
    assert result.action == "CORE_ADD" and result.size_unit == .25
    assert result.constraints["allow_new_satellite"] is False


@pytest.mark.parametrize("nominal,expected_remaining", [(1000, .01), (2000, 0), (3000, 0)])
def test_satellite_two_percent_cap_includes_existing_nominal(macro, nominal, expected_remaining):
    book, facts = base(core=False)
    add_satellite(book, facts, held=True)
    book["positions"][0]["nominal"] = nominal
    result = decide(book, facts)
    assert len(result.satellites) == 1
    row = result.satellites[0]
    assert row["suggested_nominal_cap"] == pytest.approx(expected_remaining)
    assert (row["status"] == "可执行") is (expected_remaining > 0)


@pytest.mark.parametrize("role,industry", [("CORE", "SOE_DIVIDEND"), ("SATELLITE", "SOE_DIVIDEND"),
                                         ("BALLAST", "TECHNOLOGY")])
def test_cn_accepts_only_soe_dividend_ballast(macro, role, industry):
    book, facts = base(core=False)
    book["watchlist"] = [asset("600000", role, industry, 0, market="CN")]
    result = decide(book, facts)
    assert rejection(result, "600000", "a_share_ballast_only")
    assert not result.cores and not result.satellites


@pytest.mark.parametrize("verified", [False, True])
def test_cn_requires_verified_exchange_calendar_and_reports_holiday_next_open(macro, verified):
    book, facts = base(core=False)
    book["watchlist"] = [asset("600000", "BALLAST", "SOE_DIVIDEND", 0, market="CN")]
    facts["calendars"] = {"CN": {"verified": verified, "source": "synthetic-XSHG",
                                 "sessions": ["2026-09-30", "2026-10-09"]}}
    result = decide(book, facts)
    assert not result.cores and not result.satellites
    assert rejection(result, "600000", "exchange_closed_next_open=2026-10-09" if verified
                     else "exchange_calendar_unverified")
    assert result.trace["calendars"]["CN"]["next_open"] == ("2026-10-09" if verified else None)


def test_cn_open_session_also_requires_current_csi300_membership(macro):
    book, facts = base(core=False)
    book["watchlist"] = [asset("600000", "BALLAST", "SOE_DIVIDEND", 0, market="CN")]
    facts["calendars"] = {"CN": {"verified": True, "source": "synthetic-XSHG", "sessions": [ASOF]}}
    result = decide(book, facts)
    assert rejection(result, "600000", "not_current_csi300_member")
    assert not result.satellites


def test_verified_open_cn_soe_dividend_csi300_ballast_can_be_a_satellite(macro):
    on = "2026-09-30"
    cn_days = list(pd.bdate_range(end=on, periods=300).strftime("%Y-%m-%d"))
    book, facts = base(core=False)
    book["watchlist"] = [asset("600000", "BALLAST", "SOE_DIVIDEND", 0, market="CN")]
    facts["calendars"] = {"CN": {"verified": True, "source": "synthetic-XSHG",
                                 "sessions": cn_days}}
    facts["prices"]["600000"] = price_rows(95, days=cn_days, market="CN")
    facts["memberships"]["CSI300"] = [{"symbol": "600000", "date": cn_days[0],
                                         "available_at": cn_days[0] + "T15:00:00+08:00",
                                         "source": "synthetic-CSI300", "verified": True,
                                         "member": True}]
    result = decide(book, facts, asof=on)
    assert result.action == "SATELLITE_DIP" and not result.cores
    assert len(result.satellites) == 1
    assert result.satellites[0]["symbol"] == "600000"
    assert result.satellites[0]["role"] == "BALLAST"
    assert result.satellites[0]["status"] == "可执行"


def test_promotion_refuses_old_core_not_exited_or_demoted(macro):
    book, facts = base()
    book["positions"].append(asset("BBB", "CORE",  nominal=1000, previous_role="SATELLITE"))
    book["core_promotion"] = {"symbol": "BBB", "previous_core": "AAA", "issued_on": ASOF}
    result = decide(book, facts)
    assert result.action == "OBSERVE" and result.size_unit == 0
    assert "promotion_requires_old_core_exit_or_demotion" in result.trace["book_reasons"]


def test_prior_satellite_cannot_be_promoted_without_explicit_ledger_instruction(macro):
    book, facts = base()
    book["positions"][0]["previous_role"] = "SATELLITE"
    result = decide(book, facts)
    assert result.action == "OBSERVE"
    assert "explicit_core_promotion_required" in result.trace["book_reasons"]


@pytest.mark.parametrize("old_role", [None, "SATELLITE"])
def test_explicit_promotion_is_allowed_after_old_core_exit_or_demotion(macro, old_role):
    book, facts = base()
    if old_role is None:
        book["positions"] = []
    else:
        book["positions"][0]["role"] = old_role
    book["positions"].append(asset("BBB", "CORE", nominal=1000, previous_role="SATELLITE"))
    book["core_promotion"] = {"symbol": "BBB", "previous_core": "AAA", "issued_on": ASOF}
    facts["prices"]["BBB"] = deepcopy(facts["prices"]["AAA"])
    facts["consensus"]["BBB"] = deepcopy(facts["consensus"]["AAA"])
    facts["memberships"]["SMH"].append(record(DAYS[0], symbol="BBB", member=True))
    result = decide(book, facts)
    assert result.action == "CORE_ADD" and result.size_unit == .25
    assert [row["symbol"] for row in result.cores] == ["BBB"]


@pytest.mark.parametrize("p", [.99, None])
def test_surge_attachment_is_read_only_and_never_changes_execution(macro, p):
    book, facts = base()
    decision = decide(book, facts)
    estimate = {"symbol": "AAA", "asof": ASOF, "p": p, "n": 100,
                "target": {"horizon_sessions": 1}, "nested": {"value": [1, 2]}}
    annotated = decision.with_surge([estimate])
    before = decision.to_dict()
    after = annotated.to_dict()
    assert annotated.action == decision.action == "CORE_ADD"
    assert annotated.size_unit == decision.size_unit == .25
    assert annotated.constraints == decision.constraints
    assert {k: v for k, v in after.items() if k != "surge"} == {
        k: v for k, v in before.items() if k != "surge"}
    assert decision.surge == ()
    estimate["p"] = .01
    estimate["nested"]["value"].append(3)
    assert annotated.surge[0]["p"] == p
    assert annotated.surge[0]["nested"]["value"] == (1, 2)
    with pytest.raises(TypeError):
        annotated.surge[0]["p"] = .01
    with pytest.raises(TypeError):
        annotated.constraints["allow_margin"] = False
    with pytest.raises(FrozenInstanceError):
        annotated.action = "SELL"
    json.loads(annotated.to_json())
