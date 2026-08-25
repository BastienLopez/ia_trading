"""Compare un protocole P3 seul/P3+P4 sans sélectionner une fenêtre après coup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _root(path: str | Path) -> Path:
    value = Path(path)
    return value.parent if value.name == "walk_forward_summary.json" else value


def _load_arm(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = _root(path)
    manifest_path, summary_path = root / "run_manifest.json", root / "walk_forward_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise ValueError(f"run incomplet ou non reproductible: {root}")
    return json.loads(manifest_path.read_text(encoding="utf-8")), json.loads(summary_path.read_text(encoding="utf-8"))


def _candidate_metrics(row: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in row.get("validation_candidates", []):
        if candidate.get("candidate_id") == candidate_id:
            return candidate["metrics"]
    raise ValueError(f"candidat {candidate_id} absent de la fenêtre {row.get('window')}")


def _check_manifest(manifest: dict[str, Any], protocol: dict[str, Any], p4_expected: bool) -> list[str]:
    errors = []
    if manifest.get("evaluation_mode") != protocol["evaluation_mode"]:
        errors.append("evaluation_mode différent")
    if manifest.get("transaction_fee") != protocol["transaction_fee"]:
        errors.append("transaction_fee différent")
    if manifest.get("validation_seeds") != protocol["validation_seeds"]:
        errors.append("validation_seeds différents")
    if manifest.get("candidate_ids") != [protocol["candidate_id"]]:
        errors.append("candidat différent du protocole")
    if bool(manifest.get("p4", {}).get("enabled")) is not p4_expected:
        errors.append("arm P4 incorrect")
    return errors


def compare(protocol: dict[str, Any], p3_arm: tuple[dict[str, Any], list[dict[str, Any]]],
            p4_arm: tuple[dict[str, Any], list[dict[str, Any]]]) -> dict[str, Any]:
    """Construit un verdict reproductible sur les fenêtres pré-enregistrées."""
    p3_manifest, p3_rows = p3_arm
    p4_manifest, p4_rows = p4_arm
    errors = _check_manifest(p3_manifest, protocol, False) + _check_manifest(p4_manifest, protocol, True)
    invariant_keys = ("assets", "timeframe", "candles", "initial_balance", "allow_short_cli", "short_limits_cli")
    errors.extend(f"invariant différent: {key}" for key in invariant_keys if p3_manifest.get(key) != p4_manifest.get(key))
    if p3_manifest.get("market_data_snapshot", {}).get("hashes") != p4_manifest.get("market_data_snapshot", {}).get("hashes"):
        errors.append("snapshot OHLCV différent entre P3 seul et P3+P4")
    expected = protocol["windows"]
    p3_by_window, p4_by_window = ({row.get("window"): row for row in rows} for rows in (p3_rows, p4_rows))
    if sorted(p3_by_window) != expected or sorted(p4_by_window) != expected:
        errors.append("fenêtres exécutées différentes du protocole pré-enregistré")
    if errors:
        return {"protocol": protocol["comparison_id"], "valid": False, "errors": errors, "promotion": False}

    rows = []
    for window in expected:
        baseline = _candidate_metrics(p3_by_window[window], protocol["candidate_id"])
        with_p4 = _candidate_metrics(p4_by_window[window], protocol["candidate_id"])
        base_strategy, p4_strategy = baseline["strategy"], with_p4["strategy"]
        rows.append({
            "window": window,
            "p3_return": base_strategy["total_return"],
            "p3_p4_return": p4_strategy["total_return"],
            "delta_return": p4_strategy["total_return"] - base_strategy["total_return"],
            "p4_buy_and_hold_return": with_p4["buy_and_hold"]["total_return"],
            "p4_profit_factor": with_p4["closed_trade_metrics"]["profit_factor"],
            "p4_expectancy": with_p4["closed_trade_metrics"]["expectancy"],
            "p4_above_buy_and_hold": p4_strategy["total_return"] >= with_p4["buy_and_hold"]["total_return"],
        })
    deltas = [row["delta_return"] for row in rows]
    rule = protocol["promotion_rule"]
    reasons = []
    if sum(delta > 0 for delta in deltas) < rule["minimum_p4_better_windows"]:
        reasons.append("P4 ne dépasse pas P3 sur assez de fenêtres pré-enregistrées")
    if rule["require_positive_median_delta"] and float(np.median(deltas)) <= 0:
        reasons.append("delta médian P4-P3 non positif")
    if rule["require_p4_above_buy_and_hold_median"] and float(np.median([row["p3_p4_return"] - row["p4_buy_and_hold_return"] for row in rows])) < 0:
        reasons.append("P4 reste sous Buy & Hold en médiane")
    if rule["require_positive_profit_factor_and_expectancy_each_window"] and any(
        row["p4_profit_factor"] <= 1 or row["p4_expectancy"] <= 0 for row in rows
    ):
        reasons.append("profit factor ou expectancy P4 insuffisant sur au moins une fenêtre")
    return {
        "protocol": protocol["comparison_id"], "valid": True, "promotion": not reasons,
        "promotion_rejection_reasons": reasons, "windows": rows,
        "summary": {
            "median_delta_return": float(np.median(deltas)),
            "p4_better_window_count": int(sum(delta > 0 for delta in deltas)),
            "window_count": len(rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--p3-run", required=True)
    parser.add_argument("--p3-p4-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    result = compare(protocol, _load_arm(args.p3_run), _load_arm(args.p3_p4_run))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
