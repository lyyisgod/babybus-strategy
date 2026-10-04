"""Persisted raw golden scenarios, with hand-specified rule expectations.

All prices, EPS, membership changes and statements are synthetic fixtures.
The session dates and close timestamps follow the actual XNYS calendar. No
expectation is produced by the decision engine, and no macro adapter is mocked.
"""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from macro_engine.engine import decide
from macro_engine.inputs import nyse
from macro_engine.macro import macro_facts
from macro_regime.yield_calendar import treasury_closed


FIXTURES = Path(__file__).parent / "fixtures" / "golden"
CASES_DOCUMENT = json.loads((FIXTURES / "cases.json").read_text())
CASES = CASES_DOCUMENT["cases"]
DATES = ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07",
         "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14"]
TOP_FIELDS = {"regime", "stress", "constraints", "action", "cores", "satellites",
              "rejects", "trace", "size_unit", "surge"}
STRESS_FIELDS = {"state", "author_combo", "stat_combo", "author_ixg_weak",
                 "author_move_high", "stat_ixg_weak", "stat_move_high", "pair",
                 "pair_prev", "trigger_on", "pair_streak", "clear_streak",
                 "move_pct", "move_observations", "move_proxy"}
CONSTRAINT_FIELDS = {"allow_margin", "max_gross", "max_cores", "max_industry_names",
                     "max_names", "satellite_suggested_nominal_cap",
                     "allow_new_core", "allow_new_satellite"}
BLOCKED_FIELDS = {"membership", "consensus", "five_day_gain", "not_down",
                 "industry_cap", "new_high", "deleveraging", "metal_clock"}
ACTIONS = {"OBSERVE", "HOLD", "CORE_ADD", "CORE_ADD_NO_MARGIN", "CORE_EXIT_WINDOW",
           "PRECIOUS_METALS_HOLD", "PRECIOUS_METALS_EXIT", "PRECIOUS_METALS_OBSERVE",
           "DELEVER_TO_1X", "SATELLITE_DIP", "CORE_HOLD_NO_MARGIN"}


def load_input(case):
    facts = json.loads((FIXTURES / case["raw_facts"]).read_text())
    for series, omissions in case["omit_publications"].items():
        facts["publications"][series] = [
            row for row in facts["publications"][series] if row["date"] not in omissions
        ]
    for symbol in case["omit_consensus"]:
        facts["consensus"].pop(symbol, None)
    return deepcopy(case["book"]), facts, deepcopy(case["policy_statements"])


def assert_projection(actual, expected, path="Decision"):
    """Project only declared fields, while keeping every declared value strict."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(expected) <= set(actual), path
        for key, value in expected.items():
            assert_projection(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        for index, value in enumerate(expected):
            assert_projection(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, bool) or expected is None:
        assert actual is expected, path
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=0, abs=1e-12), path
    else:
        assert actual == expected, path


def action_paths(value, path="Decision"):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "action":
                yield f"{path}.{key}"
            yield from action_paths(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from action_paths(item, f"{path}[{index}]")


def assert_structure(decision):
    output = decision.to_dict()
    assert json.loads(decision.to_json()) == output
    assert set(output) == TOP_FIELDS
    assert list(action_paths(output)) == ["Decision.action"]
    assert isinstance(output["action"], str) and output["action"] in ACTIONS
    assert set(output["stress"]) == STRESS_FIELDS
    assert isinstance(output["stress"]["state"], str)
    assert set(output["constraints"]) == CONSTRAINT_FIELDS
    assert output["constraints"]["max_cores"] == 1
    assert output["constraints"]["max_industry_names"] == 5
    assert output["constraints"]["max_names"] == 16
    assert output["constraints"]["satellite_suggested_nominal_cap"] == .02
    for field in ("allow_margin", "allow_new_core", "allow_new_satellite"):
        assert type(output["constraints"][field]) is bool
    assert output["size_unit"] in {0., .25, .5, 1.}
    if output["action"] not in {"CORE_ADD", "CORE_ADD_NO_MARGIN"}:
        assert output["size_unit"] == 0.
    for field in ("cores", "satellites", "rejects", "surge"):
        assert isinstance(output[field], list)
    assert output["surge"] == []
    assert len(output["cores"]) <= 1
    for row in output["cores"]:
        assert {"symbol", "market", "role", "industry", "status", "metrics",
                "consensus", "reason"} <= set(row)
        assert row["symbol"] == "COREX" and row["role"] == "CORE"
        assert {"ret_1d", "ret_5d", "dip_unit", "rsi14", "new_low20",
                "at_252_high", "window_start", "down_candle"} <= set(row["metrics"])
        if row["status"] == "可执行":
            assert output["action"] in {"CORE_ADD", "CORE_ADD_NO_MARGIN"}
            assert row["metrics"]["down_candle"] is True
            assert row["metrics"]["dip_unit"] == output["size_unit"]
    for row in output["satellites"]:
        assert {"symbol", "market", "role", "industry", "status", "metrics",
                "suggested_nominal_cap"} <= set(row)
        assert 0 <= row["suggested_nominal_cap"] <= .02
        if row["status"] == "可执行":
            assert output["action"] == "SATELLITE_DIP"
    for row in output["rejects"]:
        assert set(row) == {"symbol", "market", "reason"}
        assert all(isinstance(row[field], str) for field in row)
    trace = output["trace"]
    assert {"asof", "policy", "macro", "gross", "budget_ready", "book_reasons",
            "coverage", "calendars", "metal_clock", "metal_holdings", "reason",
            "computed_price_facts", "paper_n", "size_kind"} <= set(trace)
    assert set(trace["policy"]) == {"regime", "cut_index", "reason", "last_statement_date"}
    assert trace["policy"]["regime"] == output["regime"]
    assert isinstance(trace["policy"]["reason"], list)
    assert set(trace["coverage"]) == {"input_names", "cores", "satellites", "rejects", "blocked"}
    assert set(trace["coverage"]["blocked"]) == BLOCKED_FIELDS
    assert all(type(value) is int and value >= 0 for value in trace["coverage"]["blocked"].values())
    assert trace["coverage"]["cores"] == len(output["cores"])
    assert trace["coverage"]["satellites"] == len(output["satellites"])
    assert trace["coverage"]["rejects"] == len(output["rejects"])
    assert trace["calendars"]["US"] == {"verified": True, "open": True,
        "next_open": trace["asof"], "source": "exchange_calendars.XNYS"}
    assert trace["calendars"]["CN"]["verified"] is False
    assert trace["paper_n"] == 0 and trace["size_kind"] == "ordinal_not_nav_fraction"
    assert trace["macro"]["publication_coordinate"] == "shift(1) then diff(5), then backward asof"
    assert trace["macro"]["publication_units"] == "original_percentage_points_for_DGS10_and_OAS"
    if output["stress"]["state"] == "TRIGGER":
        assert output["constraints"]["allow_margin"] is False
        assert output["constraints"]["max_gross"] == 1.
    return output


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"] + "-" + case["asof"])
def test_ten_raw_golden_sessions(case):
    book, facts, policy = load_input(case)
    before = deepcopy((book, facts, policy))
    decision = decide(book, facts, policy, case["asof"])
    assert (book, facts, policy) == before
    output = assert_structure(decision)
    assert_projection(output, case["expected_projection"])
    macro = macro_facts(facts, case["asof"])
    assert macro["data_valid"] is True
    assert macro["yield_stable"] is False
    assert macro["trace"]["rejected_rows"] == []
    assert macro["stress"]["move_observations"] >= 252
    assert macro["trace"]["publication_deltas"]["dgs10_delta5"] == 0.
    assert macro["trace"]["publication_deltas"]["oas_delta5"] == 0.


def test_fixture_provenance_calendar_and_raw_contract():
    assert [case["asof"] for case in CASES] == DATES
    assert CASES_DOCUMENT["_fixture"]["synthetic"] is True
    raw = json.loads((FIXTURES / "raw_facts.json").read_text())
    assert raw["_fixture"]["synthetic"] is True
    assert raw["_fixture"]["warmup_sessions"] == 310
    calendar = nyse(2026)
    assert list(calendar.sessions_in_range(DATES[0], DATES[-1]).strftime("%Y-%m-%d")) == DATES
    forbidden = {"ret_1d", "ret_5d", "dip_unit", "stress", "at_252_high", "pressure"}
    def no_precomputed(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                no_precomputed(item)
        elif isinstance(value, list):
            for item in value:
                no_precomputed(item)
    no_precomputed(raw)
    price_dates = [row["date"] for row in raw["macro_prices"]["TLT"]]
    assert len([day for day in price_dates if day < DATES[0]]) == 310
    assert len(price_dates) == 320
    assert list(calendar.sessions_in_range(price_dates[0], DATES[-1]).strftime("%Y-%m-%d")) == price_dates
    for group in (raw["prices"], raw["macro_prices"]):
        for rows in group.values():
            assert [row["date"] for row in rows] == price_dates
            for row in rows:
                assert {"date", "adj_close", "adj_open", "source", "closed", "verified", "available_at"} <= set(row)
                assert row["source"].startswith("synthetic://golden/")
                assert row["verified"] is row["closed"] is True
                assert datetime.fromisoformat(row["available_at"]).tzinfo is not None
    dgs_dates = [row["date"] for row in raw["publications"]["DGS10"]]
    assert dgs_dates == [day for day in price_dates if not treasury_closed(pd.Timestamp(day))]
    assert "2026-10-12" not in dgs_dates
    assert all(case["expected_projection"]["size_unit"] in {0., .25, .5, 1.} for case in CASES)


def append_unavailable_evidence(facts, policy, asof):
    """Append both future dates and unavailable revisions of historical dates."""
    available = "2026-11-02T21:00:00+00:00"
    future = "2026-10-15"
    for field in ("prices", "macro_prices", "publications"):
        for rows in facts[field].values():
            # Keep the original calendar range; the backfill is an existing
            # historical observation with a later availability timestamp.
            historic = next(row for row in rows if row["date"] <= asof)
            for day in (future, historic["date"]):
                extra = deepcopy(historic)
                extra.update(date=day, available_at=available)
                if field == "publications":
                    extra["value"] = 99999.
                else:
                    extra.update(adj_close=99999., adj_open=1.)
                    if field == "macro_prices":
                        extra["close"] = 99999.
                rows.append(extra)
    for rows in facts["memberships"].values():
        for day in (future, asof):
            rows.append({"symbol": "COREX", "date": day, "member": False,
                "source": "synthetic://golden/future-membership", "verified": True,
                "available_at": available})
    # Deliberately add unavailable rows even if this case has no consensus.
    rows = facts["consensus"].setdefault("COREX", [])
    for day in (future, asof):
        rows.append({"date": day, "eps": 99999., "fiscal_period": "FY2027",
            "is_consensus": True, "source": "synthetic://golden/future-eps",
            "verified": True, "available_at": available})
    for day in (future, policy[0]["date"]):
        policy.append({"date": day, "action": "CUT", "target_range": [0., .25],
            "source": "https://www.federalreserve.gov/synthetic/golden/future.htm",
            "verified": True, "available_at": available})


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_future_append_and_unavailable_backfill_preserve_entire_decision_json(case):
    book, facts, policy = load_input(case)
    original = decide(book, facts, policy, case["asof"]).to_json()
    append_unavailable_evidence(facts, policy, case["asof"])
    appended = decide(book, facts, policy, case["asof"]).to_json()
    assert appended == original


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_missing_budget_keeps_all_rows_observation_only_and_retains_pressure_warning(case):
    book, facts, policy = load_input(case)
    book["budget"] = None
    output = assert_structure(decide(book, facts, policy, case["asof"]))
    # The unconditional >1x confirmed-pressure warning still has one account
    # action; the missing budget prevents any row from becoming executable.
    expected_action = "DELEVER_TO_1X" if case["id"] == "day06" else "OBSERVE"
    assert output["action"] == expected_action
    assert output["size_unit"] == 0.
    assert output["trace"]["budget_ready"] is False
    assert output["trace"]["execution_status"] == "观察"
    assert output["trace"]["book_reasons"] == ["missing_or_invalid_budget"]
    assert output["constraints"]["allow_new_core"] is False
    assert output["constraints"]["allow_new_satellite"] is False
    assert all(row["status"] != "可执行" for row in output["cores"] + output["satellites"])


def test_metal_clock_is_separate_from_core_hold_and_preserves_cut_number():
    first, second, third = CASES[2:5]
    expected = [(first, 1, "HOLD_FIRST_CUT", "CORE_EXIT_WINDOW"),
                (second, 2, "EXIT_SECOND_CUT", "PRECIOUS_METALS_EXIT"),
                (third, 3, "OBSERVE_THIRD_PLUS", "PRECIOUS_METALS_OBSERVE")]
    for case, cut_index, clock, action in expected:
        book, facts, policy = load_input(case)
        output = decide(book, facts, policy, case["asof"]).to_dict()
        assert output["trace"]["policy"]["cut_index"] == cut_index
        assert output["trace"]["metal_clock"] == clock
        assert output["trace"]["metal_holdings"] == ["SLV"]
        assert output["action"] == action
        assert output["trace"]["coverage"]["blocked"]["metal_clock"] == 1
        assert output["trace"]["coverage"]["satellites"] == 0
        assert len([row for row in output["rejects"] if row["symbol"] == "SLV"]) == 1
