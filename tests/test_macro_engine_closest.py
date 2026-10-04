"""Lock closest as a read-only diagnosis of the account's existing gates.

All prices, membership and consensus come from raw synthetic PIT evidence.
The macro adapter fixture only supplies an already-valid macro state, as in
the existing account contract tests; no selection gate is replaced.
"""
from copy import deepcopy
import json

import pytest

import macro_engine.engine as engine
from test_macro_engine_contract import (
    ASOF, DAYS, POLICY, add_satellite, asset, base, decide, macro, price_rows,
    record,
)


CLOSEST_FIELDS = {"symbol", "market", "role", "passed", "failed_gates", "rank_score"}
ADD_ACTIONS = {"CORE_ADD", "CORE_ADD_NO_MARGIN", "SATELLITE_DIP"}


def closest(decision):
    value = decision.to_dict()["trace"]["closest"]
    if value is not None:
        assert set(value) == CLOSEST_FIELDS
        assert isinstance(value["passed"], bool)
        assert isinstance(value["failed_gates"], list)
        assert len(value["failed_gates"]) == len(set(value["failed_gates"]))
        assert isinstance(value["rank_score"], list)
        assert len(value["rank_score"]) == 4
        assert value["rank_score"][0] == len(value["failed_gates"])
        assert value["rank_score"][1] == (0 if value["role"] == "CORE" else 1)
    return value


def add_core(book, facts, symbol, *, last=95):
    book["watchlist"].append(asset(symbol, "CORE", nominal=0))
    facts["prices"][symbol] = price_rows(last)
    facts["consensus"][symbol] = deepcopy(facts["consensus"]["AAA"])
    facts["memberships"]["SMH"].append(record(DAYS[0], symbol=symbol, member=True))


def test_all_gates_pass_closest_is_the_executable_core(macro):
    book, facts = base()
    add_satellite(book, facts)
    result = decide(book, facts)
    candidate = closest(result)
    executable = [row["symbol"] for row in result.cores + result.satellites
                  if row["status"] == "可执行"]
    assert result.action == "CORE_ADD" and result.size_unit == .25
    assert candidate["symbol"] == executable[0] == "AAA"
    assert candidate["market"] == "US" and candidate["role"] == "CORE"
    assert candidate["passed"] is True and candidate["failed_gates"] == []
    assert candidate["rank_score"][2] == -.25
    assert result.satellites[0]["status"] == "观察"


def test_no_asset_passes_closest_never_turns_a_hold_into_an_add(macro):
    book, facts = base(last=101)
    result = decide(book, facts)
    candidate = closest(result)
    assert result.action not in ADD_ACTIONS and result.size_unit == 0
    assert candidate["symbol"] == "AAA" and candidate["passed"] is False
    assert "core_not_down" in candidate["failed_gates"]
    assert not any(row["status"] == "可执行" for row in result.cores + result.satellites)
    assert result.cores[0]["reason"] == "core_not_down"


def test_empty_book_has_no_invented_closest(macro):
    book, facts = base(core=False)
    result = decide(book, facts)
    assert closest(result) is None
    assert result.to_dict()["trace"]["reason"] == ["empty_book"]
    assert result.action == "HOLD" and result.size_unit == 0
    assert result.trace["coverage"]["input_names"] == 0


def test_fact_only_stock_cannot_become_the_closest_candidate(macro):
    book, facts = base(last=101)
    facts["prices"]["UNLISTED"] = price_rows(80)
    facts["consensus"]["UNLISTED"] = deepcopy(facts["consensus"]["AAA"])
    facts["memberships"]["SMH"].append(record(DAYS[0], symbol="UNLISTED", member=True))
    result = decide(book, facts)
    assert closest(result)["symbol"] == "AAA"
    assert result.action == "HOLD"
    assert "UNLISTED" not in result.trace["computed_price_facts"]


def test_independent_failed_gates_are_all_reported_without_changing_rejects(macro):
    book, facts = base(last=130, previous=129)
    facts["memberships"]["SMH"] = []
    facts["consensus"] = {}
    result = decide(book, facts)
    candidate = closest(result)
    assert result.action == "HOLD" and result.size_unit == 0
    assert candidate["passed"] is False
    assert {"not_current_smh_member", "consensus_missing", "five_day_gain_gt_8pct",
            "core_not_down"}.issubset(candidate["failed_gates"])
    assert result.to_dict()["rejects"] == [
        {"symbol": "AAA", "market": "US", "reason": "not_current_smh_member"}]
    # Diagnostic computation does not add the skipped gates to trade coverage.
    assert result.trace["coverage"]["blocked"]["membership"] == 1
    assert result.trace["coverage"]["blocked"]["consensus"] == 0


def test_fewer_failed_gates_precedes_core_preference(macro):
    book, facts = base()
    facts["memberships"]["SMH"] = []
    add_satellite(book, facts)
    result = decide(book, facts)
    candidate = closest(result)
    assert result.action == "SATELLITE_DIP"
    assert candidate["symbol"] == "TLT" and candidate["passed"] is True
    assert candidate["failed_gates"] == []


def test_equal_gate_failures_prefer_ledger_core_over_alphabetical_satellite(macro):
    book, facts = base()
    book["positions"][0]["symbol"] = "ZZZ"
    facts["memberships"]["SMH"][0]["symbol"] = "ZZZ"
    facts["consensus"]["ZZZ"] = facts["consensus"].pop("AAA")
    facts["prices"] = {}
    book["watchlist"].append(asset("AAA", "SATELLITE", "UTILITY", 0))
    result = decide(book, facts)
    candidate = closest(result)
    assert candidate["symbol"] == "ZZZ" and candidate["role"] == "CORE"
    assert candidate["passed"] is False
    assert candidate["failed_gates"] == ["closed_price_history_missing"]
    assert result.action == "HOLD"


def test_core_tie_uses_larger_frozen_dip_ordinal(macro):
    book, facts = base()
    add_core(book, facts, "BBB", last=95)
    result = decide(book, facts)
    candidate = closest(result)
    assert result.action == "OBSERVE"
    assert candidate["symbol"] == "BBB" and candidate["passed"] is False
    assert candidate["rank_score"][2] == -.5
    assert "multiple_global_cores" in candidate["failed_gates"]


def test_equal_core_dip_uses_symbol_without_satellite_new_low_tiebreak(macro):
    book, facts = base(last=98)
    # AAA's earlier lower close makes current close not a 20-day new low.
    facts["prices"]["AAA"][-20]["adj_close"] = 97
    add_core(book, facts, "BBB", last=98)
    result = decide(book, facts)
    candidate = closest(result)
    assert result.action == "OBSERVE"
    assert candidate["symbol"] == "AAA" and candidate["rank_score"][2] == -.25
    assert candidate["passed"] is False


def test_satellite_tie_uses_lower_raw_rsi_before_symbol(macro):
    book, facts = base(core=False)
    add_satellite(book, facts, "AAA", industry="UTILITY", last=95)
    add_satellite(book, facts, "BBB", industry="UTILITY", last=90)
    for symbol in ("AAA", "BBB"):
        for index, row in enumerate(facts["prices"][symbol][-16:-1]):
            row["adj_close"] = 100. if index % 2 == 0 else 99.
    result = decide(book, facts)
    computed = result.trace["computed_price_facts"]
    assert computed["BBB"]["rsi14"] < computed["AAA"]["rsi14"]
    candidate = closest(result)
    assert result.action == "SATELLITE_DIP" and candidate["passed"] is True
    assert candidate["symbol"] == "BBB"
    assert candidate["rank_score"][2] == computed["BBB"]["rsi14"]


def test_equal_satellite_rsi_uses_twenty_day_new_low_before_symbol(macro):
    book, facts = base(core=False)
    add_satellite(book, facts, "AAA", industry="UTILITY", last=95)
    add_satellite(book, facts, "BBB", industry="UTILITY", last=95)
    for row in facts["prices"]["AAA"][-21:]:
        row["adj_close"] = 95.
    result = decide(book, facts)
    computed = result.trace["computed_price_facts"]
    assert computed["AAA"]["rsi14"] == computed["BBB"]["rsi14"] == 0
    assert computed["AAA"]["new_low20"] is False
    assert computed["BBB"]["new_low20"] is True
    candidate = closest(result)
    assert result.action == "SATELLITE_DIP" and candidate["symbol"] == "BBB"
    assert candidate["passed"] is True and candidate["rank_score"][3] == 0


def test_final_symbol_tiebreak_is_replay_stable(macro):
    book, facts = base(core=False)
    add_satellite(book, facts, "ZZZ", industry="UTILITY", last=95)
    add_satellite(book, facts, "AAA", industry="UTILITY", last=95)
    before = decide(book, facts)
    assert closest(before)["symbol"] == "AAA" and closest(before)["passed"] is True
    book["watchlist"].reverse()
    assert decide(book, facts).to_json() == before.to_json()


@pytest.mark.parametrize("missing", ["budget", "nav", "nominal"])
def test_account_evidence_failures_never_mark_closest_executable(macro, missing):
    book, facts = base()
    (book["positions"][0] if missing == "nominal" else book).pop(missing)
    result = decide(book, facts)
    candidate = closest(result)
    assert result.action == "OBSERVE" and result.size_unit == 0
    assert candidate["symbol"] == "AAA" and candidate["passed"] is False
    assert set(result.trace["book_reasons"]).issubset(candidate["failed_gates"])


def test_pressure_deleveraging_has_no_executable_closest_add(macro):
    book, facts = base()
    book["positions"][0]["nominal"] = 110000
    macro.update(pressure=True, stress={"state": "TRIGGER", "trigger_on": True})
    result = decide(book, facts)
    candidate = closest(result)
    assert result.action == "DELEVER_TO_1X" and result.size_unit == 0
    assert candidate["passed"] is False
    assert "confirmed_pressure_gross_gt_1" in candidate["failed_gates"]
    assert result.cores == result.satellites == ()


def test_pressure_below_one_still_has_an_executable_core_closest(macro):
    book, facts = base()
    macro.update(pressure=True, stress={"state": "TRIGGER", "trigger_on": True})
    result = decide(book, facts)
    candidate = closest(result)
    assert result.action == "CORE_ADD_NO_MARGIN" and result.size_unit == .25
    assert candidate["passed"] is True and candidate["failed_gates"] == []


def test_missing_macro_evidence_keeps_closest_an_observation(macro):
    book, facts = base()
    macro["data_valid"] = False
    result = decide(book, facts)
    candidate = closest(result)
    assert result.action == "OBSERVE" and candidate["passed"] is False
    assert "required_macro_evidence_or_us_session_missing" in candidate["failed_gates"]


def test_closest_remains_immutable_under_surge_attachment(macro):
    book, facts = base()
    result = decide(book, facts)
    baseline = result.to_dict()
    for p in (.99, None):
        annotated = result.with_surge([{"symbol": "AAA", "p": p}]).to_dict()
        assert annotated["trace"]["closest"] == baseline["trace"]["closest"]
        assert json.dumps({key: annotated[key] for key in ("action", "size_unit", "constraints")},
                          sort_keys=True) == json.dumps(
                              {key: baseline[key] for key in ("action", "size_unit", "constraints")},
                              sort_keys=True)
        assert annotated["rejects"] == baseline["rejects"]
    with pytest.raises(TypeError):
        result.trace["closest"]["passed"] = False
