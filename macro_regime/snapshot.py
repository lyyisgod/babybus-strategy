"""python -m macro_regime.snapshot --date YYYY-MM-DD [explicit policy/exposure]."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

from .config import POLICY_REGIMES
from .data import fetch_dataset
from .factors import DataError
from .state import DecisionInputs, decide, evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description="纯研究日频宏观状态机；政策和敞口必须由调用方提供")
    parser.add_argument("--date", required=True)
    parser.add_argument("--regime", required=True, choices=POLICY_REGIMES)
    parser.add_argument("--gross-exposure", type=float)
    parser.add_argument("--policy-file", type=Path, help='JSON: {date, gross_exposure, optional nfp_change}; --regime仍必填')
    parser.add_argument("--nfp-change", type=float, help="仅标签，单位人数")
    parser.add_argument("--core-symbol", default="SMH")
    parser.add_argument("--cache-dir", default="data/macro_regime/cache")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-stale-cache", action="store_true", help="仅允许MOVE缓存回退；仍标代理并关闭作者组合")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
            raise DataError("--date 格式必须为 YYYY-MM-DD")
        policy = {}
        if args.policy_file:
            policy = json.loads(args.policy_file.read_text())
            if not isinstance(policy, dict) or policy.get("date") != args.date:
                raise DataError("policy-file需要与--date严格匹配的完整手填快照")
        regime = args.regime
        if "policy_regime" in policy and policy["policy_regime"] != regime:
            raise DataError("policy-file中的policy_regime与显式--regime冲突")
        gross = args.gross_exposure if args.gross_exposure is not None else policy.get("gross_exposure")
        nfp = args.nfp_change if args.nfp_change is not None else policy.get("nfp_change")
        if regime not in POLICY_REGIMES or gross is None:
            raise DataError("需提供--gross-exposure或--policy-file中的敞口；--regime必须显式提供")
        gross = float(gross)
        decide(DecisionInputs(regime, "NONE", "NONE", gross, 0))
        if nfp is not None:
            nfp = float(nfp)
            if not math.isfinite(nfp):
                raise DataError("nfp_change 必须为有限标签")
        dataset = fetch_dataset(args.date, args.cache_dir, args.core_symbol, args.refresh, args.allow_stale_cache)
        result = evaluate(dataset.frame, regime, gross, args.core_symbol, nfp,
                          dgs10_publications=dataset.dgs10_publications,
                          move_publications=dataset.move_publications)
        result["data"] = dataset.metadata
        for s, level in result["author_levels"].items():
            if level["expired"]:
                dataset.metadata["warnings"].append(f"{s} author level expired as_of={level['as_of']}")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (DataError, ValueError, TypeError, OSError, ImportError) as error:
        print(f"macro_regime error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
