"""Raw-data, point-in-time and publication-calendar regression checks."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import exchange_calendars as xcals

from macro_engine.macro import macro_facts
from macro_engine import macro as macro_module
from macro_regime.config import STATISTICS
from macro_regime.yield_calendar import treasury_closed


def fixture(end="2026-10-02", count=24, *, ixg=120.0, move=130.0):
    finish = pd.Timestamp(end)
    start = finish - pd.Timedelta(days=count * 2 + 20)
    cal = xcals.get_calendar("XNYS", start=start - pd.Timedelta(days=7),
                              end=finish + pd.Timedelta(days=7))
    days = cal.sessions_in_range(start, finish)[-count:].tz_localize(None)
    facts = {"macro_prices": {}, "publications": {}}
    for name in ("IXG", "SPY", "JNK", "QQQ", "SMH", "TLT"):
        value = ixg if name == "IXG" else 100.0
        facts["macro_prices"][name] = [
            {"date": d.date().isoformat(), "close": value, "adj_close": value,
             "available_at": cal.session_close(d).isoformat(), "source": "synthetic://prices",
             "verified": True, "closed": True} for d in days
        ]
    for name in ("DGS10", "OAS", "MOVE"):
        value = move if name == "MOVE" else 4.0
        facts["publications"][name] = [
            {"date": d.date().isoformat(), "value": value,
             "available_at": cal.session_close(d).isoformat(), "source": "synthetic://publications",
             "verified": True} for d in days if name != "DGS10" or not treasury_closed(d)
        ]
    return facts


class MacroFactsTests(unittest.TestCase):
    def test_author_confirmation_and_no_decision_fields(self):
        facts = fixture()
        result = macro_facts(facts, "2026-10-02")
        self.assertEqual(result["stress"]["state"], "TRIGGER")
        self.assertTrue(result["stress"]["author_combo"])
        self.assertTrue(result["pressure"])
        self.assertTrue(result["data_valid"])
        self.assertIsNone(result["yield_stable"])
        self.assertNotIn("action", json.dumps(result["trace"]))
        self.assertEqual(result["stress"]["pair_streak"], 3)
        self.assertEqual(STATISTICS.stress_confirm_days, 2)

    def test_one_day_enter_and_two_day_release(self):
        facts = fixture(end="2026-10-06")
        first = macro_facts(facts, "2026-09-30")
        self.assertEqual(first["stress"]["state"], "UNKNOWN")
        self.assertFalse(first["pressure"])
        for row in facts["publications"]["MOVE"][-2:]:
            row["value"] = 110
        watch = macro_facts(facts, "2026-10-05")
        self.assertEqual(watch["stress"]["state"], "TRIGGER")
        self.assertEqual(watch["stress"]["clear_streak"], 1)
        release = macro_facts(facts, "2026-10-06")
        self.assertEqual(release["stress"]["state"], "WATCH")
        self.assertFalse(release["pressure"])
        self.assertEqual(release["stress"]["clear_streak"], 2)

    def test_expired_author_with_unwarmed_statistics_is_unknown(self):
        result = macro_facts(fixture(end="2027-02-01"), "2027-02-01")
        self.assertEqual(result["stress"]["state"], "UNKNOWN")
        self.assertFalse(result["stress"]["author_combo"])
        self.assertFalse(result["pressure"])
        self.assertFalse(result["data_valid"])

    def test_never_cross_assembles_author_and_statistical_legs(self):
        facts = fixture(count=280, ixg=120, move=90)
        for row in facts["macro_prices"]["IXG"][-12:]:
            row["adj_close"] = 150  # absolute weak, relative strong
        for row in facts["publications"]["MOVE"][-3:]:
            row["value"] = 110  # statistical high, absolute not high
        result = macro_facts(facts, "2026-10-02")
        self.assertTrue(result["stress"]["author_ixg_weak"])
        self.assertTrue(result["stress"]["stat_move_high"])
        self.assertFalse(result["stress"]["author_combo"])
        self.assertFalse(result["stress"]["stat_combo"])
        self.assertFalse(result["pressure"])

    def test_publication_shift_before_diff_and_ignore_derived_inputs(self):
        facts = fixture(count=12)
        for name in ("OAS", "DGS10", "MOVE"):
            for i, row in enumerate(facts["publications"][name]):
                row["value"] = 100.0 + i * 2
            facts["publications"][name][-1]["value"] = 9999
        facts.update({"stress": "NONE", "ret_5d": -999, "differentials": {"OAS": 0}})
        result = macro_facts(facts, "2026-10-02")
        self.assertEqual(result["trace"]["publication_deltas"],
                         {"oas_delta5": 10.0, "dgs10_delta5": 10.0, "move_delta5": 10.0})
        self.assertEqual(result["trace"]["returns"]["QQQ"]["r5"], 0)

    def test_future_append_and_late_price_vintages_are_inert(self):
        facts = fixture()
        baseline = macro_facts(facts, "2026-10-02")
        revised = copy.deepcopy(facts)
        for group in revised.values():
            for rows in group.values():
                future = copy.deepcopy(rows[-1])
                future["date"] = "2026-10-05"
                future["available_at"] = "2026-10-05T21:00:00+00:00"
                rows.append(future)
                unavailable = copy.deepcopy(rows[0])
                unavailable["available_at"] = "2026-10-05T21:00:00+00:00"
                unavailable["value"] = 9999
                rows.append(unavailable)
                earlier = copy.deepcopy(unavailable)
                earlier["date"] = "1950-01-03"
                rows.append(earlier)
        for rows in revised["macro_prices"].values():
            late = copy.deepcopy(rows[0])
            late.update({"close": 1.0, "adj_close": 1.0,
                         "available_at": "2026-10-02T19:00:00+00:00"})
            rows.append(late)
        self.assertEqual(baseline, macro_facts(revised, "2026-10-02"))

    def test_future_available_old_rows_do_not_expand_calendar_range(self):
        facts = fixture()
        with patch.object(macro_module, "_calendar", wraps=macro_module._calendar) as calendar:
            baseline = macro_facts(facts, "2026-10-02")
            baseline_ranges = calendar.call_args_list[:]
        revised = copy.deepcopy(facts)
        for group in revised.values():
            for rows in group.values():
                unavailable = copy.deepcopy(rows[0])
                unavailable.update({"date": "1950-01-03",
                                    "available_at": "2026-10-02T16:00:01-04:00"})
                rows.append(unavailable)
                future = copy.deepcopy(rows[0])
                future.update({"date": "2026-10-05",
                               "available_at": "2026-10-02T19:00:00+00:00"})
                rows.append(future)
        with patch.object(macro_module, "_calendar", wraps=macro_module._calendar) as calendar:
            self.assertEqual(baseline, macro_facts(revised, "2026-10-02"))
            self.assertEqual(baseline_ranges, calendar.call_args_list)

    def test_missing_oas_preserves_confirmed_pressure_and_selloff(self):
        facts = fixture()
        facts["publications"]["OAS"] = []
        for name in ("JNK", "QQQ"):
            facts["macro_prices"][name][-1]["adj_close"] = 97.0
        result = macro_facts(facts, "2026-10-02")
        self.assertFalse(result["data_valid"])
        self.assertTrue(result["pressure"])
        self.assertEqual(result["stress"]["state"], "TRIGGER")
        self.assertEqual(result["trace"]["divergence"], "CONFIRMED_SELLOFF")

    def test_missing_actual_move_proxy_without_warmup_is_unknown(self):
        facts = fixture(count=40)
        facts["publications"]["MOVE"].pop()
        result = macro_facts(facts, "2026-10-02")
        self.assertTrue(result["trace"]["move"]["proxy"])
        self.assertEqual(result["stress"]["state"], "UNKNOWN")
        self.assertFalse(result["stress"]["author_combo"])
        self.assertEqual(result["stress"]["move_observations"], 20)
        self.assertIsNone(result["stress"]["move_pct"])
        self.assertFalse(result["data_valid"])

    def test_move_gap_uses_complete_proxy_and_real_observation_count(self):
        facts = fixture(count=310)
        # One missing middle MOVE print selects the proxy for the whole span.
        del facts["publications"]["MOVE"][150]
        for i, row in enumerate(facts["macro_prices"]["IXG"]):
            row["adj_close"] = 200 - i * .1
        log_tlt = 0.0
        for i, row in enumerate(facts["macro_prices"]["TLT"]):
            log_tlt += (-1 if i % 2 else 1) * (.02 if i >= 280 else .001)
            row["adj_close"] = 100 * np.exp(log_tlt)
        result = macro_facts(facts, "2026-10-02")
        self.assertTrue(result["trace"]["move"]["proxy"])
        self.assertEqual(result["stress"]["move_observations"], 290)
        self.assertFalse(result["stress"]["author_combo"])
        self.assertTrue(result["stress"]["stat_combo"])
        self.assertTrue(result["pressure"])
        self.assertTrue(result["data_valid"])

    def test_yield_closure_validation_precedes_lag(self):
        facts = fixture(end="2026-10-13", count=280)
        # First lagged DGS10 NaN and Columbus Day must both be harmless.
        result = macro_facts(facts, "2026-10-12")
        self.assertTrue(result["data_valid"])
        self.assertEqual(result["trace"]["yield_calendar"]["dgs10_status"], "treasury_closed")
        self.assertEqual(result["trace"]["publication_deltas"]["dgs10_delta5"], 0)
        facts["publications"]["DGS10"] = [
            row for row in facts["publications"]["DGS10"] if row["date"] != "2026-10-09"
        ]
        invalid = macro_facts(facts, "2026-10-12")
        self.assertFalse(invalid["data_valid"])
        self.assertIsNone(invalid["yield_stable"])
        self.assertTrue(any("non" in reason or "非treasury_closed" in reason
                            for reason in invalid["trace"]["reasons"]))

    def test_yield_algorithm_uses_lagged_publication_sample(self):
        facts = fixture(count=280)
        for i, row in enumerate(facts["publications"]["DGS10"]):
            row["value"] = 4 + (.2 if i % 2 else -.2) if i < 260 else 4 + i * .00001
        for i, row in enumerate(facts["publications"]["MOVE"]):
            row["value"] = 300 - i * .3
        facts["publications"]["DGS10"][-1]["value"] = 99
        facts["publications"]["MOVE"][-1]["value"] = 9999
        result = macro_facts(facts, "2026-10-02")
        self.assertTrue(result["yield_stable"])
        ys = pd.Series([row["value"] for row in facts["publications"]["DGS10"]]).shift(1)
        expected = ys.diff().rolling(10, min_periods=10).std(ddof=1).iloc[-1]
        self.assertAlmostEqual(result["trace"]["yield_calendar"]["yield_std10"], expected)
        self.assertGreaterEqual(result["trace"]["yield_calendar"]["yield_low_streak"], 5)
        self.assertGreaterEqual(result["trace"]["yield_calendar"]["move_nonrise_streak"], 5)

    def test_metadata_and_latest_close_required(self):
        facts = fixture()
        facts["macro_prices"]["IXG"][-1]["verified"] = False
        facts["macro_prices"]["SPY"][-1]["closed"] = False
        facts["macro_prices"]["QQQ"][-1]["available_at"] = "2026-10-02T16:00:00"
        result = macro_facts(facts, "2026-10-02")
        self.assertFalse(result["data_valid"])
        self.assertEqual(result["stress"]["state"], "UNKNOWN")
        self.assertIsNone(result["trace"]["returns"]["QQQ"]["r5"])
        self.assertEqual(len(result["trace"]["rejected_rows"]), 3)

    def test_holiday_returns_labeled_observation(self):
        facts = fixture()
        result = macro_facts(facts, "2026-10-04")
        self.assertEqual(result["trace"]["observed_session"], "2026-10-02")
        self.assertFalse(result["trace"]["asof_is_session"])
        self.assertTrue(result["data_valid"])


if __name__ == "__main__":
    unittest.main()
