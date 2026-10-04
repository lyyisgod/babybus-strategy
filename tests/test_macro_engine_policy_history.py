"""Read-only official FOMC histories; no network or fixture regeneration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from macro_engine.policy import official_policy


FIXTURES = Path(__file__).parent / "fixtures" / "policy_history"

# Independently transcribed announcement dates, actions and percent ranges.
# A mismatch names the exact official statement; the classifier is not tuned.
OFFICIAL_ROWS = {
    "2019": [
        ("2018-12-19", "HIKE", [2.25, 2.5], "HIKING", 0),
        ("2019-01-30", "HOLD", [2.25, 2.5], "PAUSE", 0),
        ("2019-03-20", "HOLD", [2.25, 2.5], "PAUSE", 0),
        ("2019-05-01", "HOLD", [2.25, 2.5], "PAUSE", 0),
        ("2019-06-19", "HOLD", [2.25, 2.5], "PAUSE", 0),
        ("2019-07-31", "CUT", [2.0, 2.25], "FIRST_CUT", 1),
        ("2019-09-18", "CUT", [1.75, 2.0], "EASING", 2),
        ("2019-10-30", "CUT", [1.5, 1.75], "EASING", 3),
    ],
    "2024": [
        ("2023-07-26", "HIKE", [5.25, 5.5], "HIKING", 0),
        ("2023-09-20", "HOLD", [5.25, 5.5], "PAUSE", 0),
        ("2023-11-01", "HOLD", [5.25, 5.5], "PAUSE", 0),
        ("2023-12-13", "HOLD", [5.25, 5.5], "PAUSE", 0),
        ("2024-01-31", "HOLD", [5.25, 5.5], "PAUSE", 0),
        ("2024-03-20", "HOLD", [5.25, 5.5], "PAUSE", 0),
        ("2024-05-01", "HOLD", [5.25, 5.5], "PAUSE", 0),
        ("2024-06-12", "HOLD", [5.25, 5.5], "PAUSE", 0),
        ("2024-07-31", "HOLD", [5.25, 5.5], "PAUSE", 0),
        ("2024-09-18", "CUT", [4.75, 5.0], "FIRST_CUT", 1),
        ("2024-11-07", "CUT", [4.5, 4.75], "EASING", 2),
        ("2024-12-18", "CUT", [4.25, 4.5], "EASING", 3),
    ],
}

FIXTURE_HASHES = {
    "2019": "501f50cb923a6903e0fcfecc874c8384ff3adecae30549d4882b0951d62dd44b",
    "2024": "657b0f1e003eaca25fc8c11c55107d538d5771cf2b980a66eafdb1858c2840ac",
}


def load_cycle(cycle):
    return json.loads((FIXTURES / f"{cycle}.json").read_text(encoding="utf-8"))


def official_source(day):
    return ("https://www.federalreserve.gov/newsevents/pressreleases/monetary"
            f"{day.replace('-', '')}a.htm")


@pytest.mark.parametrize("cycle", ["2019", "2024"])
def test_official_fixture_dates_ranges_sources_and_bytes_are_locked(cycle):
    path = FIXTURES / f"{cycle}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE_HASHES[cycle]
    fixture = load_cycle(cycle)
    assert fixture["read_only"] is True
    statements = fixture["statements"]
    assert len(statements) == len(OFFICIAL_ROWS[cycle])
    for record, (day, action, target, _, _) in zip(
            statements, OFFICIAL_ROWS[cycle]):
        context = f"Official statement mismatch: {day} {official_source(day)}"
        assert record["date"] == day, context
        assert record["action"] == action, context
        assert record["target_range"] == target, context
        assert record["source"] == official_source(day), context
        assert record["verified"] is True, context
        assert record["target_range_text"].endswith(" percent"), context


@pytest.mark.parametrize("cycle,day,regime,cut_index", [
    (cycle, day, regime, cut_index)
    for cycle, rows in OFFICIAL_ROWS.items()
    for day, _, _, regime, cut_index in rows
], ids=[
    f"{cycle}-{day}"
    for cycle, rows in OFFICIAL_ROWS.items()
    for day, *_ in rows
])
def test_real_official_statement_classification(cycle, day, regime, cut_index):
    statements = load_cycle(cycle)["statements"]
    expected = {
        "regime": regime,
        "cut_index": cut_index,
        "reason": ["initial_hike_anchor_without_prior_range"],
        "last_statement_date": day,
    }
    context = f"Official statement mismatch: {day} {official_source(day)}"
    # The full fixture contains the later announcements. They remain inert at
    # each preceding asof, rather than relying on a pre-trimmed caller input.
    assert official_policy(statements, day) == expected, context
    assert official_policy(
        [record for record in statements if record["date"] <= day], day
    ) == expected, context


@pytest.mark.parametrize("cycle", ["2019", "2024"])
def test_classification_does_not_mutate_read_only_evidence(cycle):
    path = FIXTURES / f"{cycle}.json"
    before = path.read_bytes()
    fixture = load_cycle(cycle)
    original = deepcopy(fixture)
    for record in fixture["statements"]:
        official_policy(fixture["statements"], record["date"])
    assert fixture == original
    assert path.read_bytes() == before
