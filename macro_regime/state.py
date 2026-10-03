"""压力滞回、信用分类和唯一研究动作；完全纯函数。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from .config import AUTHOR_LEVELS, POLICY_REGIMES, STATISTICS, VERSION
from .factors import DataError, author_levels, dip_units, leap_history, numeric_factors


def pressure_combinations(author_expired, move_proxy, ixg_below_125, move_above_125,
                          ixg_spy_below_ma200, move_pct):
    """两套完整组合OR；缺失分位不是0，也不能与绝对价格交叉拼装。"""
    active = not author_expired and not move_proxy
    ax, am = active and ixg_below_125, active and move_above_125
    sx = bool(ixg_spy_below_ma200)
    sm = move_pct is not None and math.isfinite(move_pct) and move_pct > 80
    author, statistical = bool(ax and am), bool(sx and sm)
    return {"author_combo": author, "stat_combo": statistical,
            "author_ixg_weak": bool(ax), "author_move_high": bool(am),
            "stat_ixg_weak": sx, "stat_move_high": bool(sm)}


def _confirm_pressure(raw, days, pair_prev=None, trigger_on=False):
    """pair_prev是上一收盘原始pair，trigger_on是上一收盘确认状态。"""
    trigger = bool(trigger_on)
    previous = None if pair_prev is None else bool(pair_prev)
    entry = int(previous is True)
    exit_count = int(previous is False)
    rows = []
    for current in raw.to_dict("records"):
        known = current.pop("known", True)
        if not known:
            entry = exit_count = 0
            rows.append({**current, "state": None, "pair_streak": 0, "clear_streak": 0,
                         "pair": None, "pair_prev": previous, "trigger_on": trigger})
            previous = None
            continue
        pair = current["author_combo"] or current["stat_combo"]
        watch = any(current[c] for c in ("author_ixg_weak", "author_move_high", "stat_ixg_weak", "stat_move_high"))
        entry = entry + 1 if pair else 0
        exit_count = exit_count + 1 if not pair else 0
        if entry >= days:
            trigger = True
        elif exit_count >= days:
            trigger = False
        rows.append({**current, "state": "TRIGGER" if trigger else "WATCH" if watch else "NONE",
                     "pair_streak": entry, "clear_streak": exit_count,
                     "pair": pair, "pair_prev": previous, "trigger_on": trigger})
        previous = pair
    return pd.DataFrame(rows, index=raw.index)


def pressure_history(frame, factors, levels=AUTHOR_LEVELS, cfg=STATISTICS):
    ixg, move = levels["IXG"], levels["MOVE"]
    rows = []
    for d, price, vol, proxy, relative, ma, pct in zip(
        frame.index, frame.IXG, frame.MOVE, frame.move_proxy,
        factors.relative_ixg, factors.relative_ma200, factors.move_pct5y
    ):
        inactive = d.date() < ixg.as_of or d.date() < move.as_of or ixg.expired(d.date()) or move.expired(d.date())
        combos = pressure_combinations(inactive, bool(proxy), price < ixg.value, vol > move.value,
                                       relative < ma, pct)
        # 分位暖机不足仅关闭统计组合；不阻止有效作者组合确认。
        rows.append({**combos, "known": bool(np.isfinite([price, vol, relative]).all())})
    return _confirm_pressure(pd.DataFrame(rows, index=frame.index), cfg.stress_confirm_days)


def divergence(row, stress):
    """确认抛售优先，然后利率归因、未确认信用、风险偏好背离。"""
    if row.JNK_r5 < -0.015 and row.QQQ_r5 < -0.015 and stress == "TRIGGER":
        return "CONFIRMED_SELLOFF"
    if row.JNK_r5 < -0.015 and row.oas_delta5 <= 0 and row.dgs10_delta5 > 0:
        return "RATE_NOT_CREDIT"
    if (row.JNK_r5 < -0.015 or row.JNK_down_streak >= 5) and row.SMH_r5 >= -0.01:
        return "UNCONFIRMED_CREDIT"
    if row.JNK_r5 > 0 and row.QQQ_r5 < 0:
        return "RISK_ON_DIVERGENCE"
    return "NONE"


@dataclass(frozen=True)
class DecisionInputs:
    policy_regime: str
    stress: str
    divergence: str
    gross_exposure: float
    size_unit: float
    yield_stable: bool | None = False
    vix_high: bool = False


@dataclass(frozen=True)
class FactorInputs:
    """直接对账/表驱动用因子输入；生产路径与测试共享组合和动作函数。"""
    author_expired: bool
    move_proxy: bool
    ixg_below_125: bool
    move_above_125: bool
    ixg_spy_below_ma200: bool
    move_pct: float | None
    gross: float
    regime: str
    jnk_ret_5d: float
    qqq_ret_5d: float
    smh_ret_5d: float
    oas_chg_5d: float
    dgs10_up_5d: bool
    yield_stable: bool | None
    core_ret_1d: float
    core_ret_5d: float
    jnk_down_streak: int = 0
    vix_high: bool = False
    pair_prev: bool = False
    trigger_on: bool = False  # 输入为上一收盘的确认状态。


def decide_factors(inputs: FactorInputs, history=()) -> dict:
    """无history时用pair_prev/trigger_on接续；有history则从其首日接续回放。"""
    for current in (*history, inputs):
        for name in ("jnk_ret_5d", "qqq_ret_5d", "smh_ret_5d", "oas_chg_5d", "core_ret_1d", "core_ret_5d"):
            if not math.isfinite(getattr(current, name)):
                raise DataError(f"必需因子缺失: {name}")
    records = [pressure_combinations(i.author_expired, i.move_proxy, i.ixg_below_125,
                                    i.move_above_125, i.ixg_spy_below_ma200, i.move_pct)
               for i in (*history, inputs)]
    initial = history[0] if history else inputs
    stress_row = _confirm_pressure(pd.DataFrame(records), STATISTICS.stress_confirm_days,
                                   initial.pair_prev, initial.trigger_on).iloc[-1]
    row = pd.Series({"JNK_r5": inputs.jnk_ret_5d, "QQQ_r5": inputs.qqq_ret_5d,
                     "SMH_r5": inputs.smh_ret_5d, "oas_delta5": inputs.oas_chg_5d,
                     "dgs10_delta5": int(inputs.dgs10_up_5d), "JNK_down_streak": inputs.jnk_down_streak})
    div = divergence(row, stress_row.state)
    result = decide(DecisionInputs(inputs.regime, stress_row.state, div, inputs.gross,
                                  dip_units(inputs.core_ret_1d, inputs.core_ret_5d),
                                  inputs.yield_stable, inputs.vix_high))
    result.update({"author_combo": bool(stress_row.author_combo), "stat_combo": bool(stress_row.stat_combo),
                   "stress": stress_row.state, "divergence": div,
                   "pair": bool(stress_row.pair), "pair_prev": bool(stress_row.pair_prev),
                   "trigger_on": bool(stress_row.trigger_on),
                   "pair_streak": int(stress_row.pair_streak), "clear_streak": int(stress_row.clear_streak)})
    return result


def decide(inputs: DecisionInputs) -> dict:
    p, s, d = inputs.policy_regime, inputs.stress, inputs.divergence
    if p not in POLICY_REGIMES or s not in ("NONE", "WATCH", "TRIGGER"):
        raise DataError("非法政策/压力状态")
    if d not in ("NONE", "CONFIRMED_SELLOFF", "RATE_NOT_CREDIT", "UNCONFIRMED_CREDIT", "RISK_ON_DIVERGENCE"):
        raise DataError("非法信用背离状态")
    if not math.isfinite(inputs.gross_exposure) or inputs.gross_exposure < 0:
        raise DataError("gross_exposure 必须为有限非负名义敞口/NAV")
    if inputs.size_unit not in (0, 0.25, 0.5, 1):
        raise DataError("非法加仓单位")
    pressure = s == "TRIGGER" or d == "CONFIRMED_SELLOFF"
    # 该标志只反映压力约束；政策持有/退出动作仍独立限制新增。
    margin = not pressure
    constraints = {"allow_margin": margin, "max_gross": None if margin else 1.0,
                   "allow_new_satellite": p == "HIKING" and not pressure,
                   "allow_core_hold": p != "FIRST_CUT"}
    size = 0.0
    if p == "FIRST_CUT":
        action = "CORE_EXIT_WINDOW"
    elif pressure and inputs.gross_exposure > 1:
        action = "DELEVER_TO_1X"
    elif p == "HIKING" and d == "CONFIRMED_SELLOFF":
        action = "DELEVER_TO_1X"
    elif p == "HIKING" and s == "TRIGGER":
        if inputs.size_unit > 0:
            action, size = "CORE_ADD_NO_MARGIN", inputs.size_unit
        else:
            action = "CORE_HOLD_NO_MARGIN"
    elif p == "HIKING" and d in ("UNCONFIRMED_CREDIT", "RATE_NOT_CREDIT"):
        action, size = "BUY_CORE_SEMI", inputs.size_unit
    elif p == "HIKING" and d == "RISK_ON_DIVERGENCE":
        action, size = "INDEX_DIP_BUY", inputs.size_unit
    elif inputs.yield_stable is True and s != "TRIGGER" and p == "HIKING":
        action = "EARNINGS_SLEEVE"
        constraints["allow_new_satellite"] = False
    elif p == "HIKING":
        action, size = "DIP_ADD", inputs.size_unit
    elif p == "PAUSE":
        action = "CORE_HOLD"
    else:
        action = "REENTRY_WATCH" if inputs.vix_high else "HOLD"
    return {"action": action, "size_unit": float(size), "constraints": constraints}


def evaluate(frame: pd.DataFrame, policy_regime: str, gross_exposure: float,
             core_symbol="SMH", nfp_change=None, levels=AUTHOR_LEVELS, cfg=STATISTICS, *,
             dgs10_publications=None, move_publications=None):
    """只计算传入数据最后一个完整收盘；调用方负责截断未来与对齐。"""
    if policy_regime not in POLICY_REGIMES:
        raise DataError("policy_regime 必须由调用方明确提供")
    if nfp_change is not None and not math.isfinite(nfp_change):
        raise DataError("nfp_change 必须是人数单位的有限标签")
    factors = numeric_factors(frame, core_symbol, cfg,
                              dgs10_publications=dgs10_publications, move_publications=move_publications)
    row = factors.iloc[-1]
    required = [c for c in factors if not c.endswith("_new_low20")
                and not c.startswith("yield_")
                and c not in ("vix_spike", "move_pct5y", "move_q80", "move_asof",
                              "move_yield_leg", "move_age_sessions", "move_nonrise_streak",
                              "move_delta5", "dgs10_status")]
    missing = [c for c in required if not np.isfinite(row[c])]
    # 缺失MOVE分位不是必需序列缺失，仅让stat_combo=false。
    if len(frame) < cfg.stress_confirm_days or frame[["IXG", "MOVE", "IXG_adj", "SPY_adj"]].tail(cfg.stress_confirm_days).isna().any().any():
        missing.append("stress_confirmation_history")
    if missing:
        raise DataError(f"因子暖机不足或历史窗口缺失: {missing}")
    history = pressure_history(frame, factors, levels, cfg)
    stress_row = history.iloc[-1]
    stress = stress_row.state
    d = divergence(row, stress)
    units = dip_units(row[f"{core_symbol}_r1"], row[f"{core_symbol}_r5"])
    stable = None if pd.isna(row.yield_stable) else bool(row.yield_stable)
    result = decide(DecisionInputs(policy_regime, stress, d, gross_exposure, units,
                                  stable, bool(frame.VIX.iloc[-1] > row.vix_q80)))
    reasons = [f"policy_regime={policy_regime} (caller_supplied)",
               f"stress={stress}; author_combo={bool(stress_row.author_combo)}; stat_combo={bool(stress_row.stat_combo)}; pair_streak={int(stress_row.pair_streak)}; clear_streak={int(stress_row.clear_streak)}",
               f"IXG_close={frame.IXG.iloc[-1]:.6g}; MOVE_close={frame.MOVE.iloc[-1]:.6g}; move_proxy={bool(frame.move_proxy.iloc[-1])}",
               f"relative_ixg={row.relative_ixg:.6g}; relative_ma200={row.relative_ma200:.6g}; move_pct5y={row.move_pct5y:.6g}; move_q80={row.move_q80:.6g}",
               f"divergence={d}; JNK_r5={row.JNK_r5:.6g}; JNK_down_streak={row.JNK_down_streak:.6g}; QQQ_r5={row.QQQ_r5:.6g}; SMH_r5={row.SMH_r5:.6g}",
               f"dgs10_delta5={row.dgs10_delta5:.6g} percentage_points; oas_delta5={row.oas_delta5:.6g} percentage_points",
               f"yield_stable={stable}; yield_std10={row.yield_std10:.6g}; yield_std_median252={row.yield_std_median252:.6g}; move_delta5={row.move_delta5:.6g}",
               f"yield_asof={row.yield_asof}; low_streak={row.yield_low_streak}; move_asof={row.move_asof}; nonrise_streak={row.move_nonrise_streak}; dgs10_status={row.dgs10_status}",
               f"gross_exposure={gross_exposure:.6g}; core_symbol={core_symbol}; core_r1={row[f'{core_symbol}_r1']:.6g}; core_r5={row[f'{core_symbol}_r5']:.6g}; dip_units={units} (ordinal_not_NAV_fraction)",
               f"vix_spike={bool(row.vix_spike)}; VIX={frame.VIX.iloc[-1]:.6g}; vix_ma20={row.vix_ma20:.6g}; vix_delta1={row.vix_delta1:.6g}; vix_q80={row.vix_q80:.6g}"]
    if pd.isna(row.yield_std_median252):
        reasons.append("insufficient_yield_history: median requires 200 valid volatility observations in 252 publications")
    if pd.isna(row.move_yield_leg):
        reasons.append("insufficient_move_history_or_missing_asof: yield_stable=null")
    leap = leap_history(factors.QQQ_r1)
    labels = ["LEAP_2Y_0_6D"] if leap.iloc[-1] else []
    if row.QQQ_r1 <= -0.02 and not leap.iloc[-1]:
        reasons.append("LEAP_2Y_0_6D suppressed: 20 trading-session cooldown after previous emission")
    if pd.isna(row.move_pct5y):
        reasons.append(f"move_pct5y missing: move_observations={int(row.move_observations)}; minimum=252; stat_combo=false")
    if row[f"{core_symbol}_r5"] > 0.08:
        labels.append("CHASE_BLOCK")
    beta = {s: float(row[f"{s}_beta"]) for s in ("SMH", "SLV", "JNK")}
    eligibility = {s: "CORE_ELIGIBLE" if row[f"{s}_beta"] >= 0 else "RATE_SUPPRESSED"
                   for s in {"SMH", "SLV", "JNK", "TLT", core_symbol}}
    reasons += [f"hike_beta[{s}]={row[f'{s}_beta']:.6g}; eligibility={eligibility[s]}" for s in sorted(eligibility)]
    # 资格控制只约束新核心候选；不清掉调用方已经定义的核心。
    if result["action"] == "BUY_CORE_SEMI" and eligibility[core_symbol] != "CORE_ELIGIBLE":
        result["size_unit"] = 0.0
        reasons.append("new_core_blocked: hike_beta<0; existing_core_hold_allowed")
    satellites = []
    if result["action"] == "BUY_CORE_SEMI" and result["constraints"]["allow_new_satellite"]:
        for s in ("SLV", "JNK", "TLT"):
            if eligibility[s] == "RATE_SUPPRESSED" and (row[f"{s}_rsi14"] < 30 or row[f"{s}_new_low20"]):
                satellites.append({"symbol": s, "label": "SATELLITE_DIP", "max_nominal": 0.02,
                                   "limit_kind": "suggested_nominal_cap", "position_calculated": False})
                reasons.append(f"SATELLITE_DIP[{s}]: RSI14={row[f'{s}_rsi14']:.6g}; new_low20={bool(row[f'{s}_new_low20'])}; suggested_max_nominal=0.02; position_not_calculated")
    displayed = author_levels(frame, levels)
    for s, level in displayed.items():
        if level["expired"]:
            reasons.append(f"author_levels[{s}] expired as_of={level['as_of']}; statistics_unaffected")
    if nfp_change is not None:
        reasons.append(f"nfp_change={nfp_change} persons (label_only)")
    nullable_bool = lambda value: None if pd.isna(value) else bool(value)
    iso_date = lambda value: None if pd.isna(value) else value.date().isoformat()
    result.update({"version": VERSION, "date": frame.index[-1].date().isoformat(), "regime": policy_regime,
                   "stress": {"state": stress, **{c: bool(stress_row[c]) for c in ["author_combo", "stat_combo", "author_ixg_weak", "author_move_high", "stat_ixg_weak", "stat_move_high"]},
                              "pair_streak": int(stress_row.pair_streak), "clear_streak": int(stress_row.clear_streak),
                              "pair": bool(stress_row.pair), "pair_prev": bool(stress_row.pair_prev),
                              "trigger_on": bool(stress_row.trigger_on),
                              "move_pct": None if pd.isna(row.move_pct5y) else float(row.move_pct5y),
                              "move_observations": int(row.move_observations)},
                   "divergence": d, "yield_stable": stable,
                   "yield_calendar": {"yield_asof": iso_date(row.yield_asof),
                                      "move_asof": iso_date(row.move_asof),
                                      "yield_leg": nullable_bool(row.yield_leg),
                                      "move_leg": nullable_bool(row.move_yield_leg),
                                      "yield_low_streak": int(row.yield_low_streak),
                                      "move_nonrise_streak": None if pd.isna(row.move_nonrise_streak) else int(row.move_nonrise_streak),
                                      "dgs10_status": row.dgs10_status},
                   "balance_sheet_stress": stress == "TRIGGER" and row.dgs10_delta5 > 0 and row.oas_delta5 > 0,
                   "hike_beta": beta, "beta_unit": "decimal_return_per_yield_percentage_point",
                   "core_symbol": core_symbol, "eligibility": eligibility, "satellite_candidates": satellites,
                   "vix_spike": bool(row.vix_spike), "labels": labels, "nfp_change": nfp_change,
                   "reason": reasons, "author_levels": displayed})
    result["balance_sheet_stress"] = bool(result["balance_sheet_stress"])
    return result
