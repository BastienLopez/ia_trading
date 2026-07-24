"""Audit reproductible des campagnes P3 multi-actifs, strictement sur validation.

Le script ne lance aucun entraînement et ne lit jamais les artefacts ``test``.
Il consolide les métriques par seed et par fenêtre, applique les garde-fous du
runner, puis construit un classement robuste et déterministe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_trading.scripts.run_real_multi_asset_walk_forward import _aggregate_validation, _eligible


EXPECTED_ARTIFACTS = (
    "metrics.json", "equity_curve.csv", "actions.csv", "trades.csv", "closed_trades.csv", "trade_audit.json",
)
MANUAL_SHORTLISTS = {0: [1], 1: [0, 1, 3], 2: [], 3: [3, 5], 4: [4, 6, 7]}
FATAL_LOG_PATTERN = re.compile(r"traceback|timed out|\bkilled\b|exception", re.IGNORECASE)


class AnalysisError(RuntimeError):
    """Artefacts P3 invalides ou incomplets."""


def parameter_fingerprint(parameters: dict[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _number(value: Any, label: str, *, allow_inf: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label}: valeur numérique invalide") from exc
    if math.isnan(number) or (math.isinf(number) and not allow_inf):
        raise AnalysisError(f"{label}: valeur NaN ou infinie")
    return number


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile)) if values else float("nan")


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)), "min": float(array.min()),
        "max": float(array.max()), "std": float(array.std(ddof=0)), "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p25": float(np.percentile(array, 25)),
    }


def _result_roots(run_dirs: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for run_dir in run_dirs:
        if (run_dir / "walk_forward_summary.json").is_file():
            roots.append(run_dir)
            continue
        roots.extend(sorted(path.parent for path in run_dir.rglob("walk_forward_summary.json")))
    unique = {path.resolve() for path in roots}
    if not unique:
        raise AnalysisError("aucun walk_forward_summary.json trouvé")
    return sorted(unique, key=lambda path: str(path))


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"JSON illisible: {path}") from exc


def _candidate_from_summary(summary_row: dict[str, Any], slot: int) -> dict[str, Any]:
    candidates = summary_row.get("validation_candidates")
    if not isinstance(candidates, list) or slot >= len(candidates):
        raise AnalysisError("summary incohérent avec les dossiers candidats")
    summary_candidate = candidates[slot]
    parameters = summary_candidate.get("parameters")
    if not isinstance(parameters, dict):
        raise AnalysisError("paramètres candidat absents du summary")
    original_index = int(summary_candidate.get("original_candidate_index", slot))
    fingerprint = parameter_fingerprint(parameters)
    stored_fingerprint = summary_candidate.get("parameter_fingerprint")
    if stored_fingerprint is not None and stored_fingerprint != fingerprint:
        raise AnalysisError(f"empreinte incohérente dans le summary pour le candidat {slot}")
    return {
        "candidate_id": summary_candidate.get("candidate_id", f"p3-candidate-{original_index:02d}-{fingerprint}"),
        "original_candidate_index": original_index,
        "parameter_fingerprint": fingerprint,
        "parameters": parameters,
    }


def _validate_seed(
    directory: Path, candidate: dict[str, Any], run_name: str, window: int, seed: int,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    for filename in EXPECTED_ARTIFACTS:
        if not (directory / filename).is_file():
            issues.append(f"artefact manquant: {filename}")
    if issues:
        return {}, issues
    try:
        metrics = _read_json(directory / "metrics.json")
        audit = _read_json(directory / "trade_audit.json")
        equity = pd.read_csv(directory / "equity_curve.csv")
        actions = pd.read_csv(directory / "actions.csv")
        closed = pd.read_csv(directory / "closed_trades.csv")
        trades_path = directory / "trades.csv"
        # Le runner historique sérialise un DataFrame sans colonnes en fichier
        # vide lorsqu'aucun fill n'a été exécuté. C'est une politique inactive,
        # pas un CSV corrompu; les garde-fous de trades l'écarteront ensuite.
        try:
            trades = pd.DataFrame() if trades_path.stat().st_size == 0 else pd.read_csv(trades_path)
        except pd.errors.EmptyDataError:
            trades = pd.DataFrame()
    except (AnalysisError, OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return {}, [str(exc)]
    if not isinstance(metrics, dict) or not isinstance(audit, dict):
        return {}, ["metrics ou audit non objet JSON"]
    try:
        strategy = metrics["strategy"]
        benchmark = metrics["buy_and_hold"]
        ledger = metrics["closed_trade_metrics"]
        diversity = metrics["action_diversity"]
        for key in ("total_return", "annual_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown"):
            _number(strategy.get(key), f"strategy.{key}")
        _number(benchmark.get("total_return"), "buy_and_hold.total_return")
        for key in ("closed_trade_count", "expectancy", "win_rate"):
            _number(ledger.get(key), f"closed_trade_metrics.{key}")
        profit_factor = _number(ledger.get("profit_factor"), "closed_trade_metrics.profit_factor", allow_inf=True)
        if not isinstance(diversity.get("direction_counts"), dict):
            raise AnalysisError("action_diversity.direction_counts absent")
        if equity.empty or actions.empty:
            raise AnalysisError("equity_curve.csv ou actions.csv vide")
        if not np.isfinite(equity.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
            raise AnalysisError("equity_curve.csv contient NaN ou infini")
        if not bool(audit.get("reconciled")):
            raise AnalysisError("ledger non réconcilié")
        reconciliation_error = _number(audit.get("reconciliation_error"), "trade_audit.reconciliation_error")
        if abs(reconciliation_error) > 1e-6:
            raise AnalysisError("erreur de réconciliation supérieure à 1e-6")
        initial = _number(audit.get("initial_balance"), "trade_audit.initial_balance")
        ending = _number(audit.get("ending_equity"), "trade_audit.ending_equity")
        if not np.isclose(float(equity["agent_rl"].iloc[-1]), ending, atol=1e-6, rtol=1e-9):
            raise AnalysisError("equity finale incohérente avec trade_audit")
        if not np.isclose(ending / initial - 1.0, float(strategy["total_return"]), atol=1e-6, rtol=1e-8):
            raise AnalysisError("rendement stratégie incohérent avec equity")
        if int(audit.get("closed_trades", -1)) != len(closed):
            raise AnalysisError("nombre de closed_trades incohérent avec le ledger")
        if int(ledger["closed_trade_count"]) != len(closed):
            raise AnalysisError("métrique closed_trade_count incohérente avec closed_trades.csv")
    except (AnalysisError, KeyError, TypeError, ValueError) as exc:
        return {}, [str(exc)]
    metadata = metrics.get("candidate")
    if metadata and (
        metadata.get("parameter_fingerprint") != candidate["parameter_fingerprint"]
        or metadata.get("parameters") != candidate["parameters"]
    ):
        issues.append("identité candidat incohérente entre metrics et summary")
    if math.isinf(profit_factor):
        issues.append("profit_factor infini neutralisé dans le classement")
    observation = {
        "run": run_name, "window": window, "seed": seed, "directory": str(directory),
        "metrics": metrics, "audit": audit, "issues": issues,
        "excess_return": float(metrics.get("strategy", {}).get("total_return", float("nan")))
        - float(metrics.get("buy_and_hold", {}).get("total_return", float("nan"))),
        "profit_factor": profit_factor, "trades_empty": trades.empty,
    }
    return observation, issues


def _aggregate_window(observations: list[dict[str, Any]], min_closed_trades: int, max_order_rate: float) -> tuple[dict[str, Any], bool, str | None]:
    metrics_by_seed = [item["metrics"] for item in observations]
    aggregate = _aggregate_validation(metrics_by_seed)
    eligible, reason = _eligible(aggregate, min_closed_trades, max_order_rate)
    return aggregate, eligible, reason


def _rank_scale(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) <= 1 or ordered[0][1] == ordered[-1][1]:
        return {key: 0.5 for key in values}
    result: dict[str, float] = {}
    for position, (key, _) in enumerate(ordered):
        result[key] = position / (len(ordered) - 1)
    return result


def _summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    observations = candidate["observations"]
    excess = [item["excess_return"] for item in observations]
    returns = [float(item["metrics"]["strategy"]["total_return"]) for item in observations]
    drawdowns = [float(item["metrics"]["strategy"]["max_drawdown"]) for item in observations]
    sharpes = [float(item["metrics"]["strategy"]["sharpe_ratio"]) for item in observations]
    sortinos = [float(item["metrics"]["strategy"]["sortino_ratio"]) for item in observations]
    calmars = [float(item["metrics"]["strategy"]["calmar_ratio"]) for item in observations]
    closed = [float(item["metrics"]["closed_trade_metrics"]["closed_trade_count"]) for item in observations]
    expectancy = [float(item["metrics"]["closed_trade_metrics"]["expectancy"]) for item in observations]
    finite_pf = [item["profit_factor"] for item in observations if math.isfinite(item["profit_factor"]) and item["metrics"]["closed_trade_metrics"]["closed_trade_count"] >= 20]
    windows = candidate["windows"]
    candidate.update({
        "observation_count": len(observations),
        "seed_benchmark_rate": float(np.mean(np.asarray(excess) > 0.0)),
        "windows_beating_benchmark": sum(window["aggregate"]["strategy"]["total_return"] > window["aggregate"]["buy_and_hold"]["total_return"] for window in windows),
        "windows_passing_guards": sum(window["eligible"] for window in windows),
        "excess": _stats(excess), "return": _stats(returns), "drawdown": _stats(drawdowns),
        "sharpe": _stats(sharpes), "sortino": _stats(sortinos), "calmar": _stats(calmars),
        "closed_trades": _stats(closed), "expectancy": _stats(expectancy),
        "profit_factor_robust": float(np.median(np.minimum(finite_pf, 3.0))) if finite_pf else 0.0,
        "guard_failures": sorted({window["reason"] for window in windows if window["reason"]}),
    })
    return candidate


def _score_candidates(candidates: list[dict[str, Any]]) -> None:
    fields = {
        "guards": {item["candidate_id"]: item["windows_passing_guards"] / max(1, len(item["windows"])) for item in candidates},
        "benchmark": {item["candidate_id"]: item["seed_benchmark_rate"] for item in candidates},
        "median_excess": {item["candidate_id"]: item["excess"]["median"] for item in candidates},
        "p25_excess": {item["candidate_id"]: item["excess"]["p25"] for item in candidates},
        "worst_excess": {item["candidate_id"]: item["excess"]["min"] for item in candidates},
        "drawdown": {item["candidate_id"]: -item["drawdown"]["max"] for item in candidates},
        "sharpe": {item["candidate_id"]: item["sharpe"]["median"] for item in candidates},
        "profit_factor": {item["candidate_id"]: item["profit_factor_robust"] for item in candidates},
        "stability": {item["candidate_id"]: -item["return"]["std"] for item in candidates},
        "trades": {item["candidate_id"]: min(1.0, item["closed_trades"]["median"] / 20.0) for item in candidates},
    }
    scaled = {name: _rank_scale(values) for name, values in fields.items()}
    profiles = {
        "conservateur": {"guards": .28, "benchmark": .16, "p25_excess": .18, "worst_excess": .14, "drawdown": .10, "stability": .08, "trades": .06},
        "equilibre": {"guards": .23, "benchmark": .15, "median_excess": .15, "p25_excess": .12, "worst_excess": .08, "drawdown": .09, "sharpe": .07, "profit_factor": .05, "stability": .03, "trades": .03},
        "performance": {"guards": .18, "benchmark": .14, "median_excess": .24, "p25_excess": .10, "worst_excess": .08, "drawdown": .07, "sharpe": .09, "profit_factor": .05, "stability": .02, "trades": .03},
    }
    for candidate in candidates:
        identifier = candidate["candidate_id"]
        candidate["scores"] = {
            profile: float(sum(weight * scaled[field][identifier] for field, weight in weights.items()))
            for profile, weights in profiles.items()
        }
        candidate["robust_score"] = float(np.mean(list(candidate["scores"].values())))
    for profile in (*profiles, "robust_score"):
        ordered = sorted(candidates, key=lambda item: (-item["scores"][profile] if profile != "robust_score" else -item[profile], item["candidate_id"]))
        for rank, candidate in enumerate(ordered, start=1):
            candidate.setdefault("ranks", {})[profile] = rank
    for candidate in candidates:
        rank_values = list(candidate["ranks"].values())
        candidate["rank_spread"] = max(rank_values) - min(rank_values)


def analyze_runs(
    run_dirs: list[str | Path], *, expected_seeds: tuple[int, ...] = (42, 314, 2024),
    min_closed_trades: int = 20, max_agent_order_rate: float = 0.35,
) -> dict[str, Any]:
    roots = _result_roots([Path(path) for path in run_dirs])
    grouped: dict[str, dict[str, Any]] = {}
    anomalies: list[dict[str, str]] = []
    source_rows: list[dict[str, Any]] = []
    index_fingerprints: dict[int, set[str]] = defaultdict(set)

    for root in roots:
        summary_path = root / "walk_forward_summary.json"
        summary = _read_json(summary_path)
        if not isinstance(summary, list) or len(summary) != 1 or not isinstance(summary[0], dict):
            raise AnalysisError(f"summary doit contenir une seule fenêtre: {summary_path}")
        summary_row = summary[0]
        window = int(summary_row["window"])
        run_name = root.parent.name
        log_path = root.parent / "command.log"
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if FATAL_LOG_PATTERN.search(line):
                    anomalies.append({"severity": "error", "source": str(log_path), "message": line.strip()})
        validation_root = root / f"window_{window:02d}" / "validation"
        slots = sorted((path for path in validation_root.glob("candidate_*") if path.is_dir()), key=lambda path: path.name)
        expected_candidates = summary_row.get("validation_candidates")
        if not isinstance(expected_candidates, list) or not expected_candidates:
            raise AnalysisError(f"validation_candidates absent ou vide: {summary_path}")
        if len(slots) != len(expected_candidates):
            anomalies.append({"severity": "error", "source": str(validation_root), "message": f"{len(slots)} candidats trouvés, {len(expected_candidates)} attendus selon le summary"})
        source_rows.append({"run": run_name, "window": window, "root": str(root), "summary_hash": _hash_file(summary_path), "command_log": str(log_path)})
        for slot_path in slots:
            match = re.fullmatch(r"candidate_(\d+)", slot_path.name)
            if not match:
                continue
            slot = int(match.group(1))
            candidate = _candidate_from_summary(summary_row, slot)
            index_fingerprints[candidate["original_candidate_index"]].add(candidate["parameter_fingerprint"])
            global_id = f"p3-{candidate['parameter_fingerprint']}"
            entry = grouped.setdefault(global_id, {
                "candidate_id": global_id, "parameter_fingerprint": candidate["parameter_fingerprint"],
                "parameters": candidate["parameters"], "original_indices": set(), "observations": [], "windows": [],
            })
            if entry["parameters"] != candidate["parameters"]:
                anomalies.append({"severity": "error", "source": str(slot_path), "message": "même empreinte avec paramètres différents"})
            entry["original_indices"].add(candidate["original_candidate_index"])
            seed_dirs = {int(path.name.removeprefix("seed_")): path for path in slot_path.glob("seed_*") if path.is_dir() and path.name.removeprefix("seed_").isdigit()}
            if set(seed_dirs) != set(expected_seeds):
                anomalies.append({"severity": "error", "source": str(slot_path), "message": f"seeds trouvées {sorted(seed_dirs)}, attendues {list(expected_seeds)}"})
            window_observations = []
            for seed in expected_seeds:
                seed_path = seed_dirs.get(seed)
                if seed_path is None:
                    continue
                observation, issues = _validate_seed(seed_path, candidate, run_name, window, seed)
                if issues:
                    for issue in issues:
                        severity = "warning" if issue == "profit_factor infini neutralisé dans le classement" else "error"
                        anomalies.append({"severity": severity, "source": str(seed_path), "message": issue})
                if observation:
                    entry["observations"].append(observation)
                    window_observations.append(observation)
            if len(window_observations) == len(expected_seeds):
                aggregate, eligible, reason = _aggregate_window(window_observations, min_closed_trades, max_agent_order_rate)
                summary_metrics = summary_row["validation_candidates"][slot].get("metrics", {})
                if aggregate and not np.isclose(aggregate["strategy"]["total_return"], float(summary_metrics.get("strategy", {}).get("total_return", np.nan)), atol=1e-10):
                    anomalies.append({"severity": "error", "source": str(summary_path), "message": f"summary et seeds divergent pour candidate_{slot:02d}"})
                entry["windows"].append({"run": run_name, "window": window, "eligible": eligible, "reason": reason, "aggregate": aggregate, "observations": window_observations})

    for index, fingerprints in sorted(index_fingerprints.items()):
        if len(fingerprints) > 1:
            anomalies.append({"severity": "warning", "source": "identity", "message": f"index {index} désigne plusieurs empreintes: {sorted(fingerprints)}"})
    candidates = [_summarize_candidate(item) for item in grouped.values() if item["observations"]]
    if not candidates:
        raise AnalysisError("aucun candidat analysable")
    _score_candidates(candidates)
    candidates.sort(key=lambda item: (-item["robust_score"], item["candidate_id"]))
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        candidate["original_indices"] = sorted(candidate["original_indices"])
    per_window: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        for window in candidate["windows"]:
            per_window[(window["run"], window["window"])].append({"candidate": candidate, "window": window})
    run_rankings = []
    for (run_name, window), items in sorted(per_window.items(), key=lambda item: item[0][1]):
        ranked = sorted(
            items,
            key=lambda item: (
                not item["window"]["eligible"],
                -item["window"]["aggregate"]["strategy"]["total_return"] + item["window"]["aggregate"]["buy_and_hold"]["total_return"],
                item["candidate"]["candidate_id"],
            ),
        )
        run_rankings.append({"run": run_name, "window": window, "ranking": [item["candidate"]["candidate_id"] for item in ranked]})
    return {
        "stage": "initial_top5", "expected_seeds": list(expected_seeds), "sources": source_rows,
        "anomalies": anomalies, "candidates": candidates, "run_rankings": run_rankings,
        "global_status": "PROVISIONAL" if not any(item["windows_passing_guards"] == len(item["windows"]) for item in candidates) else "PARTIAL",
        "methodology": {
            "official_guards": "_aggregate_validation + _eligible du runner, puis intégrité des artefacts et ledger réconcilié",
            "scores": "moyenne des profils conservateur, équilibré et performance; PF plafonné à 3 et neutralisé sous 20 clôtures",
        },
    }


def evidence_first_top5(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Shortlist de retest : validation officielle puis constance seed par seed."""
    def key(candidate: dict[str, Any]) -> tuple[float, ...]:
        all_seed_windows = sum(
            all(observation["excess_return"] > 0.0 for observation in window["observations"])
            for window in candidate["windows"]
        )
        best_passing_excess = max(
            (window["aggregate"]["seed_stability"]["median_excess_return"] for window in candidate["windows"] if window["eligible"]),
            default=float("-inf"),
        )
        return (
            candidate["windows_passing_guards"], all_seed_windows,
            int(candidate["closed_trades"]["median"] >= 20.0), candidate["seed_benchmark_rate"],
            best_passing_excess, candidate["excess"]["p25"], candidate["candidate_id"],
        )

    # L'identifiant inverse le dernier critère pour conserver un ordre total stable.
    return sorted(analysis["candidates"], key=key, reverse=True)[:5]


def top5_manifest(analysis: dict[str, Any], selection_mode: str = "robust") -> dict[str, Any]:
    candidates = evidence_first_top5(analysis) if selection_mode == "evidence-first" else analysis["candidates"][:5]
    return {
        "schema_version": 1,
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "original_candidate_index": candidate["original_indices"][0],
                "parameter_fingerprint": candidate["parameter_fingerprint"],
                "parameters": candidate["parameters"],
                "source_runs": sorted({observation["run"] for observation in candidate["observations"]}),
                "original_seeds": analysis["expected_seeds"],
            }
            for candidate in candidates
        ],
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_report(analysis: dict[str, Any], manifest_path: str, selection_mode: str = "robust") -> str:
    lines = [
        "# Phase 3 - Top 5 provisoire", "",
        f"**Statut global : {analysis['global_status']} - AUCUN CANDIDAT GLOBALEMENT VALIDÉ.**",
        "", f"Runs analysés : {len(analysis['sources'])}; configurations uniques : {len(analysis['candidates'])}; observations run x seed : {sum(item['observation_count'] for item in analysis['candidates'])}.",
        "", "## Sources", "",
        "| Run | Fenêtre | Racine résultat | Hash summary |", "|---|---:|---|---|",
    ]
    for source in analysis["sources"]:
        lines.append(f"| {source['run']} | {source['window']} | `{source['root']}` | `{source['summary_hash']}` |")
    lines += ["", f"Seeds initiaux communs : `{analysis['expected_seeds']}`. Les cinq campagnes étaient en validation; aucun artefact `test` n'a été lu.", "", "## Gardes et score", "", "Gardes officielles : rendement net strictement supérieur au benchmark sur les 3 seeds, diversité directionnelle, achat et vente présents, au moins 20 clôtures, PF > 1, expectancy > 0 et turnover <= 35 %. L'analyse refuse aussi ledger non réconcilié, fichiers manquants et métriques non finies.", "", "Score robuste : moyenne de trois profils. Conservateur privilégie gardes, P25/pire excès et drawdown; équilibré privilégie gardes, proportion de seeds gagnantes et médiane; performance privilégie médiane. Le PF est plafonné à 3 et vaut 0 sous 20 clôtures.", "", "## Comparaison avec l'analyse manuelle", "", "| Fenêtre | Candidats manuels | Résultat Codex | Accord | Explication |", "|---:|---|---|---|---|"]
    by_window = {item["window"]: item for item in analysis["run_rankings"]}
    for window, manual in MANUAL_SHORTLISTS.items():
        ranking = by_window.get(window, {}).get("ranking", [])
        winner = next((candidate for candidate in analysis["candidates"] if candidate["candidate_id"] == ranking[0]), None) if ranking else None
        winner_index = winner["original_indices"] if winner else []
        agreement = "accord" if winner and any(index in manual for index in winner_index) else "désaccord"
        winner_window = next((entry for entry in winner["windows"] if entry["window"] == window), None) if winner else None
        manual_candidates = [
            candidate for candidate in analysis["candidates"]
            if any(index in manual for index in candidate["original_indices"])
        ]
        manual_windows = [
            (candidate, next((entry for entry in candidate["windows"] if entry["window"] == window), None))
            for candidate in manual_candidates
        ]
        manual_windows = [(candidate, entry) for candidate, entry in manual_windows if entry]
        best_manual = max(
            manual_windows,
            key=lambda item: item[1]["aggregate"]["seed_stability"]["median_excess_return"],
            default=(None, None),
        )
        if winner_window is None:
            explanation = "fenêtre absente"
        elif agreement == "accord":
            explanation = f"candidat {winner_index}: excès médian {_pct(winner_window['aggregate']['seed_stability']['median_excess_return'])}; garde {'PASS' if winner_window['eligible'] else 'FAIL'}"
        elif best_manual[1] is not None:
            explanation = (
                f"Codex {winner_index}: {_pct(winner_window['aggregate']['seed_stability']['median_excess_return'])}; "
                f"meilleur manuel {best_manual[0]['original_indices']}: {_pct(best_manual[1]['aggregate']['seed_stability']['median_excess_return'])}"
            )
        else:
            explanation = f"Codex {winner_index}: excès médian {_pct(winner_window['aggregate']['seed_stability']['median_excess_return'])}"
        lines.append(f"| {window} | {manual or 'aucun'} | {winner_index or 'aucun'} | {agreement} | {explanation} |")
    lines += ["", "## Classement par run", "", "| Fenêtre | Rang | Candidat Codex | Empreinte | Gardes fenêtre | Excès médian (3 seeds) |", "|---:|---:|---|---|---|---:|"]
    for item in analysis["run_rankings"]:
        for rank, candidate_id in enumerate(item["ranking"][:3], start=1):
            candidate = next(candidate for candidate in analysis["candidates"] if candidate["candidate_id"] == candidate_id)
            window = next(window for window in candidate["windows"] if window["window"] == item["window"] and window["run"] == item["run"])
            lines.append(f"| {item['window']} | {rank} | {candidate['original_indices']} / {candidate['candidate_id']} | `{candidate['parameter_fingerprint']}` | {'PASS' if window['eligible'] else 'FAIL: ' + str(window['reason'])} | {_pct(window['aggregate']['seed_stability']['median_excess_return'])} |")
    lines += ["", "## Détails des gagnants par fenêtre", ""]
    for item in analysis["run_rankings"]:
        candidate = next(candidate for candidate in analysis["candidates"] if candidate["candidate_id"] == item["ranking"][0])
        lines += [f"### Fenêtre {item['window']} - {candidate['candidate_id']}", "", f"Index d'origine : {candidate['original_indices']}; empreinte `{candidate['parameter_fingerprint']}`.", "", "```json", json.dumps(candidate["parameters"], indent=2, sort_keys=True), "```", "", "| Seed | Rendement | B&H | Excès | PF | Expectancy | DD | Clôtures | Diversité |", "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for observation in sorted((entry for entry in candidate["observations"] if entry["window"] == item["window"]), key=lambda entry: entry["seed"]):
            metrics = observation["metrics"]
            lines.append(f"| {observation['seed']} | {_pct(metrics['strategy']['total_return'])} | {_pct(metrics['buy_and_hold']['total_return'])} | {_pct(observation['excess_return'])} | {observation['profit_factor']:.2f} | {metrics['closed_trade_metrics']['expectancy']:.2f} | {_pct(metrics['strategy']['max_drawdown'])} | {metrics['closed_trade_metrics']['closed_trade_count']} | {'PASS' if metrics['action_diversity']['passed'] and metrics['action_diversity']['directional_passed'] else 'FAIL'} |")
        lines += [""]
    lines += ["", "## Top 5 global", "", "| Rang | ID stable | Index | Agent | Empreinte | Fenêtres PASS | Seeds > B&H | Médiane excès | P25 | Pire | DD méd./max | PF robuste | Trades méd. | Score | Statut |", "|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
    for candidate in analysis["candidates"][:5]:
        status = "PROVISOIRE" if candidate["windows_passing_guards"] < len(candidate["windows"]) else "EXPLOITABLE"
        lines.append(f"| {candidate['rank']} | `{candidate['candidate_id']}` | {candidate['original_indices']} | {candidate['parameters']['agent_type']} | `{candidate['parameter_fingerprint']}` | {candidate['windows_passing_guards']}/{len(candidate['windows'])} | {_pct(candidate['seed_benchmark_rate'])} | {_pct(candidate['excess']['median'])} | {_pct(candidate['excess']['p25'])} | {_pct(candidate['excess']['min'])} | {_pct(candidate['drawdown']['median'])} / {_pct(candidate['drawdown']['max'])} | {candidate['profit_factor_robust']:.2f} | {candidate['closed_trades']['median']:.1f} | {candidate['robust_score']:.3f} | {status} |")
    lines += ["", "## Fiches des cinq candidats", ""]
    for candidate in analysis["candidates"][:5]:
        lines += [f"### {candidate['rank']}. {candidate['candidate_id']}", "", f"Origine : index {candidate['original_indices']}; empreinte `{candidate['parameter_fingerprint']}`; seeds `{analysis['expected_seeds']}`.", "", "```json", json.dumps(candidate["parameters"], indent=2, sort_keys=True), "```", "", "| Run | Fenêtre | Seed | Rendement | B&H | Excès | DD | PF | Clôtures | Diversité | Audit |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for observation in sorted(candidate["observations"], key=lambda item: (item["window"], item["seed"])):
            metrics = observation["metrics"]
            lines.append(f"| {observation['run']} | {observation['window']} | {observation['seed']} | {_pct(metrics['strategy']['total_return'])} | {_pct(metrics['buy_and_hold']['total_return'])} | {_pct(observation['excess_return'])} | {_pct(metrics['strategy']['max_drawdown'])} | {observation['profit_factor']:.2f} | {metrics['closed_trade_metrics']['closed_trade_count']} | {'PASS' if metrics['action_diversity']['passed'] and metrics['action_diversity']['directional_passed'] else 'FAIL'} | {'OK' if observation['audit']['reconciled'] else 'FAIL'} |")
        lines += ["", f"Forces : médiane d'excès {_pct(candidate['excess']['median'])}, seeds gagnantes {_pct(candidate['seed_benchmark_rate'])}.", f"Faiblesses/gardes : {', '.join(candidate['guard_failures']) or 'aucune sur les fenêtres observées'}.", f"Provenance : {', '.join('`' + observation['directory'] + '`' for observation in candidate['observations'])}.", ""]
    lines += ["## Candidats non retenus", ""]
    for candidate in analysis["candidates"][5:]:
        lines.append(f"- `{candidate['candidate_id']}` : score {candidate['robust_score']:.3f}; gardes échouées : {', '.join(candidate['guard_failures']) or 'classement inférieur' }.")
    if selection_mode == "evidence-first":
        lines += ["", "## Sélection corrigée pour le retest", "", "Cette sélection remplace le top 5 de score pour le prochain tour. Ordre : fenêtres passant les gardes, fenêtres où les trois seeds battent le benchmark, médiane d'au moins 20 clôtures, puis proportion totale de seeds gagnantes.", "", "| Rang retest | Candidat | Index | Fenêtres PASS | Fenêtres 3/3 > B&H | Trades médians | Seeds > B&H | Motif |", "|---:|---|---|---:|---:|---:|---:|---|"]
        for rank, candidate in enumerate(evidence_first_top5(analysis), start=1):
            all_seed_windows = sum(all(observation["excess_return"] > 0.0 for observation in window["observations"]) for window in candidate["windows"])
            lines.append(f"| {rank} | `{candidate['candidate_id']}` | {candidate['original_indices']} | {candidate['windows_passing_guards']} | {all_seed_windows} | {candidate['closed_trades']['median']:.1f} | {_pct(candidate['seed_benchmark_rate'])} | {'validation officielle disponible' if candidate['windows_passing_guards'] else 'retest exploratoire, pas validé'} |")
    lines += ["", "## Protocole suivant", "", f"Le manifeste exact des cinq configurations est `{manifest_path}`. Deux fenêtres entièrement nouvelles et complètes sont actuellement disponibles avec `--days 1800` : 5 et 6, avec les seeds nouvelles communes `[73, 211, 997]`. Une troisième fenêtre indépendante n'est pas disponible avant de nouvelles bougies closes : la créer en décalant artificiellement `--days` recouvrirait des validations déjà utilisées. Les deux commandes sûres sont dans `run_top5_available_windows.ps1`; aucun test hors-échantillon ne sera exécuté.", "", "## Anomalies", ""]
    if analysis["anomalies"]:
        for anomaly in analysis["anomalies"]:
            lines.append(f"- **{anomaly['severity'].upper()}** `{anomaly['source']}` : {anomaly['message']}")
    else:
        lines.append("Aucune anomalie d'artefact détectée.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", required=True, help="Dossiers de campagnes ou racines horodatées à analyser.")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[42, 314, 2024])
    parser.add_argument("--min-closed-trades", type=int, default=20)
    parser.add_argument("--max-agent-order-rate", type=float, default=0.35)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--top5-config", required=True)
    parser.add_argument("--selection-mode", choices=("robust", "evidence-first"), default="robust")
    args = parser.parse_args()
    try:
        analysis = analyze_runs(
            args.run_dir, expected_seeds=tuple(args.expected_seeds), min_closed_trades=args.min_closed_trades,
            max_agent_order_rate=args.max_agent_order_rate,
        )
    except AnalysisError as exc:
        print(f"analyse P3 invalide: {exc}", file=sys.stderr)
        return 2
    analysis["selection_mode"] = args.selection_mode
    manifest = top5_manifest(analysis, args.selection_mode)
    Path(args.output_json).write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    Path(args.top5_config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.top5_config).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(args.report).write_text(render_report(analysis, args.top5_config, args.selection_mode), encoding="utf-8")
    errors = [item for item in analysis["anomalies"] if item["severity"] == "error"]
    if errors:
        print(f"analyse P3 terminée avec {len(errors)} anomalie(s) bloquante(s)", file=sys.stderr)
        return 2
    print(f"analyse P3 terminée: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
