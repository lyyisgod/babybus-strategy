"""纯选择层，只消费宏观 JSON 与有出处的股票事实，不导入宏观引擎。"""
from copy import deepcopy
from datetime import datetime
import math
from zoneinfo import ZoneInfo


def available_by_close(available, day, market):
    if market == "US":
        import exchange_calendars as xc
        import pandas as pd
        on = pd.Timestamp(day)
        cal = xc.get_calendar("XNYS", start=on - pd.Timedelta(days=10), end=on + pd.Timedelta(days=10))
        return cal.is_session(on) and available <= cal.session_close(on).to_pydatetime()
    close = datetime.fromisoformat(day + "T15:00:00").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return available <= close


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def gate(macro):
    stress = (macro.get("stress") or {}).get("state")
    regime = macro.get("regime", "UNKNOWN")
    pressure = stress == "TRIGGER" or macro.get("divergence") == "CONFIRMED_SELLOFF"
    valid = macro.get("data_valid") is True and stress in {"NONE", "WATCH", "PENDING", "TRIGGER"}
    out = {"allow_new_core": False, "allow_new_satellite": False,
           "existing_core_only": False, "max_core_per_market": 1,
           "max_satellites": 3, "satellite_suggested_nominal_cap": .02,
           "allow_margin": (macro.get("constraints") or {}).get("allow_margin"),
           "max_gross": (macro.get("constraints") or {}).get("max_gross"),
           "action": "HOLD", "reason": []}
    if not valid:
        out["reason"] = ["required_macro_data_missing"]
    elif regime == "UNKNOWN":
        out["reason"] = ["official_regime_unknown"]
    elif stress == "PENDING" and macro.get("critical_null"):
        out["reason"] = ["pending_with_critical_null"]
    elif regime == "FIRST_CUT":
        out.update(action="CORE_EXIT_WINDOW", reason=["first_cut_overrides_prices"])
    elif regime == "EASING":
        out["reason"] = ["easing_no_new_long"]
    elif pressure:
        out.update(allow_margin=False, max_gross=1.0, existing_core_only=True,
                   reason=["pressure_existing_core_only"])
    elif regime in {"HIKING", "PAUSE"}:
        out.update(allow_new_core=True, allow_new_satellite=macro.get("yield_stable") is not True,
                   reason=["earnings_revisions_required"] if macro.get("yield_stable") is True else [])
        if regime == "PAUSE":
            out["allow_margin"] = False
    else:
        out["reason"] = ["invalid_regime"]
    return out


def value(candidate, name):
    """每个数值必须带日期、序列名和来源；拒绝过期/未来/盘中事实。"""
    fact = candidate.get("facts", {}).get(name, {})
    if (fact.get("date") != candidate["asof"] or not fact.get("series")
            or not fact.get("source") or fact.get("closed") is not True):
        return None
    v = fact.get("value")
    return v if finite(v) or isinstance(v, bool) else None


def fundamental(candidate):
    """同一预测财年、63 个当地交易日端点、当时已知一致预期。"""
    c = candidate.get("consensus", {})
    fields = ("baseline_eps", "current_eps")
    if not all(finite(c.get(k)) for k in fields) or c["baseline_eps"] <= 0:
        return None, None, None, "consensus_missing_or_nonpositive_baseline"
    if (c.get("baseline_date") != candidate.get("window_start") or c.get("date") != candidate["asof"]
            or not c.get("source") or not c.get("series") or not c.get("is_consensus")
            or not c.get("fiscal_period") or c.get("baseline_fiscal_period") != c["fiscal_period"]):
        return None, None, None, "consensus_period_or_window_unverified"
    try:
        for key, cutoff in (("baseline_available_at", candidate["window_start"]),
                            ("available_at", candidate["asof"])):
            available = datetime.fromisoformat(c[key])
            if available.tzinfo is None or not available_by_close(available, cutoff, candidate["market"]):
                raise ValueError()
    except (KeyError, ValueError, TypeError):
        return None, None, None, "consensus_availability_unverified"
    revision = c["current_eps"] / c["baseline_eps"] - 1
    downgrade = max(0., -revision)
    drawdown = value(candidate, "drawdown_63d")
    if drawdown is None:
        return None, None, revision, "drawdown_missing"
    broken = downgrade > .05 and not math.isclose(downgrade, .05, rel_tol=0., abs_tol=1e-12)
    return not broken, drawdown - downgrade, revision, (
        "eps_downgrade_gt_5pct" if broken else "")


def membership(candidate, universe):
    item = candidate.get("membership", {})
    return item.get("universe") == universe and item.get("date") == candidate["asof"] and bool(item.get("source"))


def select(macro, candidates, *, calendars, gross_exposure=None):
    """无仓位计算：core size 是宏观模块的序数，卫星 .02 只是名义上限。

    主分相同仅接受调用方带出处的 secondary_score；仍并列则不擅自选名。
    卫星成长/股息模板分别排序，以交替轮选满足总数上限，不合成分数。
    """
    permissions = gate(macro)
    result = {"macro": deepcopy(macro), "gate": permissions, "action": permissions["action"],
              "size_unit": 0., "core": {},
              "satellites": [], "rows": [], "position_calculated": False}
    if permissions["reason"] in (["required_macro_data_missing"], ["official_regime_unknown"], ["pending_with_critical_null"]):
        result["stop_reason"] = permissions["reason"]
        return result  # 不消费候选；宏观缺失时不能硬选。
    core_pools, satellite_pools = {}, {"growth": [], "dividend": []}
    pressure = permissions["existing_core_only"]
    for c in candidates:
        market, symbol, role = c["market"], c["symbol"], c["role"]
        calendar = calendars.get(market, {})
        r = {"market": market, "symbol": symbol, "role": role, "asof": c["asof"],
             "mispricing_score": None, "dividend_premium": None, "fundamental_gate": None,
             "eps_revision": None, "ma50_broken": value(c, "below_ma50"),
             "ma200_broken": value(c, "below_ma200"), "ret_1d": value(c, "ret_1d"),
             "ret_5d": value(c, "ret_5d"), "macro_gate": deepcopy(permissions),
             "status": "删除", "action": "HOLD", "size_unit": 0., "reason": [],
             "facts": deepcopy(c.get("facts", {})), "consensus": deepcopy(c.get("consensus"))}
        result["rows"].append(r)
        def reject(reason):
            r["reason"].append(reason)
        if role not in {"核心", "卫星"} or market not in {"US", "CN"}:
            reject("unsupported_role_or_market"); continue
        if c["asof"] != calendar.get("last_close") or not calendar.get("source"):
            reject("market_calendar_or_close_date_unverified"); continue
        if market == "US" and c["asof"] != macro["asof"]:
            reject("price_macro_date_mismatch"); continue
        if r["ret_1d"] is None or r["ret_5d"] is None:
            reject("closed_price_history_missing"); continue
        if r["ret_5d"] > .08:
            reject("five_day_gain_gt_8pct"); continue
        if macro["regime"] in {"FIRST_CUT", "EASING"}:
            r["action"] = permissions["action"]
            reject("policy_no_new_long"); continue
        if pressure and (role != "核心" or c.get("existing_core") is not True):
            reject("pressure_blocks_new_core_and_satellite"); continue
        if role == "核心" and pressure:
            # 已有核心不因新核心候选筛选而强制清仓。
            pass
        else:
            good, score, revision, failure = fundamental(c)
            r.update(fundamental_gate=good, mispricing_score=score, eps_revision=revision)
            if role == "核心":
                if not permissions["allow_new_core"]:
                    reject("new_core_disabled"); continue
                if good is not True:
                    reject(failure); continue
                if market == "US":
                    if not membership(c, "SMH"):
                        reject("not_current_smh_holding"); continue
                    dd = value(c, "drawdown_63d")
                    if not (r["ma50_broken"] is True or (dd is not None and dd >= .10)):
                        reject("no_core_price_candidate"); continue
                else:
                    if not membership(c, "CSI300"):
                        reject("not_current_csi300_member"); continue
                    roe, growth, dd = (value(c, k) for k in ("roe", "revenue_growth", "drawdown_63d"))
                    if any(x is None for x in (roe, growth, dd)) or not (roe > 0 and growth > 0 and dd >= .15):
                        reject("csi300_financial_or_drawdown_gate_failed"); continue
                    if macro["regime"] == "HIKING" and macro.get("factors", {}).get("dgs10_up_5d") is True:
                        high, sensitive = value(c, "high_valuation"), value(c, "financing_sensitive")
                        if high is None or sensitive is None:
                            reject("valuation_financing_classification_missing"); continue
                        if high and sensitive:
                            score -= .05
                            r["mispricing_score"] = score
                            r["reason"].append("hiking_yield_up_penalty_5pp")
                if score <= 0:
                    reject("nonpositive_mispricing_score"); continue
                if macro.get("yield_stable") is True:
                    r["reason"].append("earnings_revision_template_not_oversold_alone")
            else:
                if market != "US" or not permissions["allow_new_satellite"]:
                    reject("satellite_disabled"); continue
                beta = value(c, "hike_beta")
                smh_check = c.get("smh_membership", {})
                outside_smh = (smh_check.get("member") is False
                               and smh_check.get("date") == c["asof"] and bool(smh_check.get("source")))
                allowed = symbol in {"SLV", "SILJ"} or membership(c, "XLU") or (
                    not membership(c, "SMH") and outside_smh
                    and beta is not None and beta < 0 and value(c, "liquid") is True)
                if not allowed:
                    reject("satellite_universe_gate_failed"); continue
                rr = value(c, "rsi14")
                if not ((rr is not None and rr < 30) or value(c, "new_low20") is True):
                    reject("not_oversold"); continue
                template = c.get("template")
                if template == "growth":
                    if good is not True or score is None or score <= 0:
                        reject(failure or "nonpositive_mispricing_score"); continue
                elif template == "dividend":
                    dy = value(c, "dividend_yield")
                    treasury = macro.get("factors", {}).get("dgs10")
                    if value(c, "mature_company") is not True or dy is None or not finite(treasury):
                        reject("dividend_template_missing"); continue
                    premium = dy - treasury / 100
                    r["dividend_premium"] = premium
                    r["mispricing_score"] = None
                    if premium <= 0:
                        reject("nonpositive_dividend_premium"); continue
                else:
                    reject("valuation_template_missing"); continue
        r["status"] = "候选"
        if role == "核心":
            core_pools.setdefault(market, []).append((c, r))
        else:
            satellite_pools[c["template"]].append((c, r))
    chosen = []
    for market, pool in core_pools.items():
        if pressure:
            # 无当前错杀分就不强行排序多个已有核心。
            finalists = pool if len(pool) == 1 else []
        else:
            best = max(r["mispricing_score"] for _, r in pool)
            finalists = [(c, r) for c, r in pool if math.isclose(r["mispricing_score"], best, abs_tol=1e-12)]
            if len(finalists) > 1:
                secondary = [(value(c, "secondary_score"), c, r) for c, r in finalists]
                if all(v is not None for v, _, _ in secondary):
                    maximum = max(v for v, _, _ in secondary)
                    finalists = [(c, r) for v, c, r in secondary if v == maximum]
        if len(finalists) == 1:
            c, r = finalists[0]
            chosen.append((c, r)); result["core"][market] = c["symbol"]
        for c, r in pool:
            if not any(r is selected for _, selected in chosen):
                r["status"] = "删除"; r["reason"].append("lower_rank_or_unresolved_numeric_tie")
    for template, pool in satellite_pools.items():
        field = "mispricing_score" if template == "growth" else "dividend_premium"
        pool.sort(key=lambda pair: (-pair[1][field], pair[0]["symbol"]))
    while len(result["satellites"]) < 3 and any(satellite_pools.values()):
        for pool in satellite_pools.values():
            if pool and len(result["satellites"]) < 3:
                c, r = pool.pop(0); chosen.append((c, r)); result["satellites"].append(c["symbol"])
    for pool in satellite_pools.values():
        for _, r in pool:
            r["status"] = "删除"; r["reason"].append("satellite_count_cap")
    for c, r in chosen:
        cal = calendars[c["market"]]
        if cal.get("open_for_target") is not True:
            r.update(status="观察", action="HOLD", reason=r["reason"] + ["exchange_closed", "next_open=" + str(cal.get("next_open"))]); continue
        if c["role"] == "卫星":
            r.update(status="可执行", action="SATELLITE_DIP", suggested_nominal_cap=.02,
                     position_calculated=False); continue
        if finite(gross_exposure) and pressure and gross_exposure > 1:
            r.update(status="删除", action="DELEVER_TO_1X", reason=r["reason"] + ["gross_gt_1_before_add"]); continue
        if r["ret_1d"] >= 0:
            r.update(status="等阴线", action="HOLD", reason=r["reason"] + ["core_not_down_on_t"]); continue
        rule = macro.get("asset_rules", {}).get(c["symbol"], {})
        if rule.get("asof") != c["asof"] or rule.get("size_unit") not in {0., .25, .5, 1.}:
            r.update(status="观察", action="HOLD", reason=r["reason"] + ["macro_asset_sizing_missing"]); continue
        unit = rule["size_unit"]
        if unit == 0:
            r.update(status="观察", action="HOLD", reason=r["reason"] + ["below_add_tier"]); continue
        if pressure and not finite(gross_exposure):
            r.update(status="观察", action="HOLD", reason=r["reason"] + ["gross_exposure_required_before_pressure_add"]); continue
        r.update(status="可执行", action="CORE_ADD_NO_MARGIN" if pressure else "CORE_ADD",
                 size_unit=unit, size_kind="ordinal_not_nav_fraction", nominal_amount=None,
                 reason=r["reason"] + ["core_budget_not_provided_position_not_calculated"])
    return result
