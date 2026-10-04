"""Synthetic official-policy sequences; these tests are not investment tests."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from macro_engine.policy import official_policy


def sequence(actions):
    """Construct internally consistent percentage ranges for action strings."""
    lower = 4.0
    records = []
    start = date(2020, 1, 1)
    for index, action in enumerate(actions):
        lower += {"HIKE": .25, "HOLD": 0, "CUT": -.25}[action]
        records.append({
            "date": (start + timedelta(days=index * 30)).isoformat(),
            "action": action,
            "target_range": [lower, lower + .25],
            "source": f"https://www.federalreserve.gov/newsevents/pressreleases/{index}.htm",
            "verified": True,
        })
    return records


@pytest.mark.parametrize("actions,regime,cut_index", [
    (["HIKE"], "HIKING", 0),
    (["HIKE", "HIKE", "HIKE"], "HIKING", 0),
    (["HIKE", "HOLD"], "PAUSE", 0),
    (["HIKE"] + ["HOLD"] * 15, "PAUSE", 0),
    (["HIKE"] + ["HOLD"] * 15 + ["CUT"], "FIRST_CUT", 1),
    (["HIKE", "CUT", "CUT"], "EASING", 2),
    (["HIKE", "CUT", "HOLD", "CUT", "CUT"], "EASING", 3),
    (["HIKE"] + ["CUT"] * 7, "EASING", 7),
    (["HIKE", "CUT", "HOLD"], "EASING", 1),
    (["HIKE", "CUT", "CUT", "HOLD", "HOLD"], "EASING", 2),
    (["HIKE", "CUT", "CUT", "HIKE"], "HIKING", 0),
    (["HIKE", "CUT", "HIKE", "HOLD"], "PAUSE", 0),
    (["HIKE", "CUT", "CUT", "HIKE", "HOLD", "CUT"], "FIRST_CUT", 1),
])
def test_full_cycle_exact_state_and_cut_index(actions, regime, cut_index):
    records = sequence(actions)
    expected = {
        "regime": regime,
        "cut_index": cut_index,
        "reason": ["initial_hike_anchor_without_prior_range"],
        "last_statement_date": records[-1]["date"],
    }
    assert official_policy(records, records[-1]["date"]) == expected
    assert official_policy(list(reversed(records)), records[-1]["date"]) == expected


@pytest.mark.parametrize("actions", [
    ["HOLD"], ["CUT"], ["CUT", "CUT"], ["HOLD", "CUT", "HOLD"],
])
def test_history_without_hike_does_not_invent_cut_index(actions):
    records = sequence(actions)
    assert official_policy(records, records[-1]["date"]) == {
        "regime": "UNKNOWN", "cut_index": None,
        "reason": ["missing_hike_anchor"],
        "last_statement_date": records[-1]["date"],
    }


def test_new_hike_proves_current_cycle_after_unanchored_old_cut():
    records = sequence(["CUT", "HOLD", "HIKE", "CUT"])
    assert official_policy(records, records[-1]["date"]) == {
        "regime": "FIRST_CUT", "cut_index": 1, "reason": [],
        "last_statement_date": records[-1]["date"],
    }


@pytest.mark.parametrize("action,target", [
    ("HIKE", [4.25, 4.50]),   # unchanged range is not a hike
    ("HIKE", [4.00, 4.25]),   # falling range is not a hike
    ("HIKE", [4.25, 4.75]),   # only the upper endpoint rises
    ("CUT", [4.25, 4.50]),
    ("CUT", [4.50, 4.75]),
    ("CUT", [4.00, 4.50]),   # only the lower endpoint falls
    ("HOLD", [4.50, 4.75]),
    ("HOLD", [4.00, 4.25]),
    ("HOLD", [4.25, 4.75]),
])
def test_range_endpoints_must_both_agree_with_action(action, target):
    records = sequence(["HIKE", action])
    records[-1]["target_range"] = target
    assert official_policy(records, records[-1]["date"]) == {
        "regime": "UNKNOWN", "cut_index": None,
        "reason": ["contradictory_statements"],
        "last_statement_date": records[-1]["date"],
    }


@pytest.mark.parametrize("field,value,reason", [
    ("source", None, "missing_official_source"),
    ("source", "http://www.federalreserve.gov/statement", "nonofficial_statement"),
    ("source", "https://federalreserve.gov.example/statement", "nonofficial_statement"),
    ("source", "https://www.cmegroup.com/fedwatch", "nonofficial_statement"),
    ("source", "https://[broken", "nonofficial_statement"),
    ("verified", False, "unverified_statement"),
    ("verified", 1, "unverified_statement"),
    ("verified", None, "unverified_statement"),
    ("action", "UNKNOWN", "invalid_statement_action"),
    ("action", ["HIKE"], "invalid_statement_action"),
    ("target_range", [4.25], "invalid_target_range"),
    ("target_range", [4.5, 4.25], "invalid_target_range"),
    ("target_range", [4.5, 4.5], "invalid_target_range"),
    ("target_range", [4.25, float("nan")], "invalid_target_range"),
    ("target_range", [4.25, float("inf")], "invalid_target_range"),
    ("target_range", [True, 4.5], "invalid_target_range"),
    ("target_range", ["4.25", "4.5"], "invalid_target_range"),
])
def test_invalid_past_evidence_closes_classification(field, value, reason):
    records = sequence(["HIKE", "HOLD"])
    records[0][field] = value
    assert official_policy(records, records[-1]["date"]) == {
        "regime": "UNKNOWN", "cut_index": None, "reason": [reason],
        "last_statement_date": records[-1]["date"],
    }


@pytest.mark.parametrize("field", ["source", "verified", "action", "target_range"])
def test_missing_required_fields_are_rejected(field):
    records = sequence(["HIKE"])
    del records[0][field]
    expected_reason = {
        "source": "missing_official_source", "verified": "unverified_statement",
        "action": "invalid_statement_action", "target_range": "invalid_target_range",
    }[field]
    assert official_policy(records, records[-1]["date"]) == {
        "regime": "UNKNOWN", "cut_index": None, "reason": [expected_reason],
        "last_statement_date": records[-1]["date"],
    }


@pytest.mark.parametrize("field,value", [
    ("action", "CUT"), ("target_range", [4.50, 4.75]),
])
def test_conflicting_same_date_is_rejected(field, value):
    records = sequence(["HIKE", "HOLD"])
    conflict = deepcopy(records[-1])
    conflict[field] = value
    assert official_policy(records + [conflict], records[-1]["date"]) == {
        "regime": "UNKNOWN", "cut_index": None,
        "reason": ["contradictory_statements"],
        "last_statement_date": records[-1]["date"],
    }


def test_identical_same_date_is_not_counted_as_another_cut():
    records = sequence(["HIKE", "CUT", "CUT"])
    assert official_policy(records + [deepcopy(records[-1])], records[-1]["date"]) == {
        "regime": "EASING", "cut_index": 2,
        "reason": ["initial_hike_anchor_without_prior_range"],
        "last_statement_date": records[-1]["date"],
    }


@pytest.mark.parametrize("future", [
    {"date": "2030-01-01"},
    {"date": "2030-01-01", "action": "INVALID", "target_range": [None],
     "source": "bad", "verified": False},
    {"date": "2030-01-01", "action": "CUT", "target_range": [50, 51],
     "source": "https://www.federalreserve.gov/future", "verified": True},
])
def test_invalid_future_append_cannot_change_historical_output(future):
    records = sequence(["HIKE", "HOLD", "CUT"])
    cutoff = records[-1]["date"]
    baseline = official_policy(records, cutoff)
    assert baseline["regime"] == "FIRST_CUT" and baseline["cut_index"] == 1
    assert official_policy(records + [future], cutoff) == baseline
    assert official_policy([future] + records, cutoff) == baseline


def test_every_historical_cutoff_is_invariant_to_complete_future_history():
    records = sequence(["HIKE", "HIKE", "HOLD", "HOLD", "CUT", "HOLD",
                        "CUT", "CUT", "HIKE", "CUT"])
    for index, record in enumerate(records):
        assert official_policy(records, record["date"]) == official_policy(
            records[:index + 1], record["date"])


@pytest.mark.parametrize("asof", ["2026-10-02", "2026-10-03", "2026-10-04"])
def test_unavailable_policy_backfill_is_inert_even_before_other_validation(asof):
    records = sequence(["HIKE", "HOLD"])
    baseline = official_policy(records, asof)
    # An unavailable backfill with a historical date conflicts with the anchor;
    # another has no usable date or required fields at all.
    backfill = {"date": records[0]["date"], "action": "CUT",
                "available_at": "2026-10-02T20:01:00+00:00"}
    future = {"date": "bad", "available_at": "2026-10-05T19:00:00+00:00"}
    assert official_policy(records + [backfill, future], asof) == baseline


@pytest.mark.parametrize("available", [None, "bad", "2026-10-02T19:59:00"])
def test_optional_available_at_must_be_valid_and_timezone_aware(available):
    records = sequence(["HIKE"])
    records[0]["available_at"] = available
    assert official_policy(records, "2026-10-02") == {
        "regime": "UNKNOWN", "cut_index": None,
        "reason": ["invalid_statement_available_at"], "last_statement_date": None,
    }


def test_available_at_exact_session_close_is_inclusive():
    records = sequence(["HIKE"])
    records[0]["available_at"] = "2026-10-02T20:00:00+00:00"
    assert official_policy(records, "2026-10-02")["regime"] == "HIKING"


@pytest.mark.parametrize("records,asof,reason", [
    (None, "2026-10-02", "missing_official_statements"),
    ([], "2026-10-02", "no_statements_on_or_before_asof"),
    ([{"date": "2030-01-01"}], "2026-10-02", "no_statements_on_or_before_asof"),
    ([{"date": "invalid"}], "2026-10-02", "invalid_statement_date"),
    ([{}], "2026-10-02", "invalid_statement_date"),
    ([], "invalid", "invalid_asof"),
])
def test_empty_or_undated_material_has_explicit_unknown_reason(records, asof, reason):
    assert official_policy(records, asof) == {
        "regime": "UNKNOWN", "cut_index": None, "reason": [reason],
        "last_statement_date": None,
    }
