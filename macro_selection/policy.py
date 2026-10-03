"""官方政策证据契约；不接受价格或 DFF 作为政策分类输入。"""
from datetime import datetime
import math
from urllib.parse import urlparse


def official_regime(statements, probabilities, asof):
    """原文抽取记录由采集者提交；不完整/矛盾时关闭分类。

    statements 按时间包含官方记录，至少最近两次；cut_count 是仅由这
    两次能证明的下限。前一次暂停而本次降息，无法证明是不是第一刀。
    """
    unknown = {"regime": "UNKNOWN", "regime_source": "official"}
    try:
        if len(statements) != 2 or not probabilities:
            raise ValueError("missing_official_policy_material")
        older, latest = sorted(statements, key=lambda x: x["date"])
        for record in (older, latest):
            host = urlparse(record["source"]).hostname
            if host not in {"federalreserve.gov", "www.federalreserve.gov"}:
                raise ValueError("nonofficial_statement")
            if record["date"] > asof or record["action"] not in {"HIKE", "HOLD", "CUT"}:
                raise ValueError("future_or_invalid_statement")
            if not record.get("original_text") or not record.get("verified"):
                raise ValueError("unverified_statement")
            low, high = record["target_range_pct"]
            if not all(math.isfinite(x) for x in (low, high)) or low >= high:
                raise ValueError("invalid_target_range")
        if older["date"] >= latest["date"]:
            raise ValueError("duplicate_meeting")
        lo_diff = latest["target_range_pct"][0] - older["target_range_pct"][0]
        hi_diff = latest["target_range_pct"][1] - older["target_range_pct"][1]
        action = latest["action"]
        if not ((action == "HIKE" and lo_diff > 0 and hi_diff > 0)
                or (action == "CUT" and lo_diff < 0 and hi_diff < 0)
                or (action == "HOLD" and lo_diff == hi_diff == 0)):
            raise ValueError("contradictory_statements")
        if probabilities["source_name"] != "CME FedWatch" or not probabilities.get("verified"):
            raise ValueError("unverified_futures_probabilities")
        if urlparse(probabilities["source"]).hostname not in {"www.cmegroup.com", "cmegroup.com"}:
            raise ValueError("nonofficial_futures_source")
        if not (latest["date"] < probabilities["meeting_date"] and asof < probabilities["meeting_date"]):
            raise ValueError("wrong_next_meeting")
        observed = datetime.fromisoformat(probabilities["data_asof"])
        if observed.tzinfo is None or probabilities["market_date"] != asof:
            raise ValueError("stale_or_undated_probabilities")
        p = probabilities["probabilities"]
        if set(p) != {"cut", "hold", "hike"} or not all(math.isfinite(x) and 0 <= x <= 1 for x in p.values()):
            raise ValueError("invalid_probabilities")
        if abs(sum(p.values()) - 1) > .001:
            raise ValueError("probabilities_do_not_sum_to_one")
        regime = "UNKNOWN"
        if action == "HIKE":
            regime = "HIKING"
        elif action == "HOLD" and older["action"] == "HIKE" and p["cut"] < .5:
            regime = "PAUSE"
        elif action == "CUT" and older["action"] == "HIKE":
            regime = "FIRST_CUT"
        elif action == "CUT" and older["action"] == "CUT":
            regime = "EASING"
        return {"regime": regime, "regime_source": "official",
                "reason": [] if regime != "UNKNOWN" else ["cycle_not_provable_from_last_two_statements"]}
    except (KeyError, TypeError, ValueError) as error:
        return {**unknown, "reason": [str(error)]}
