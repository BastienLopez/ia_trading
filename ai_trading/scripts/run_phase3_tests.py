"""Exécute les tests Phase 3 séquentiellement, un fichier par processus.

Certaines extensions GPU (Torch/Triton) peuvent planter si des familles de
modèles incompatibles sont chargées dans le même interpréteur. Cette commande
ne parallélise rien : elle isole chaque fichier de test, conserve la sortie
pytest et retourne un code non nul dès qu'au moins un fichier échoue.

À lancer dans le conteneur :
``python -m ai_trading.scripts.run_phase3_tests``.
"""

from __future__ import annotations

import subprocess
import sys
import signal
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_GROUPS = (
    "ai_trading/tests/rl/test_*.py",
    "ai_trading/rl/tests/test_*.py",
    "ai_trading/tests/risk/test_*.py",
    "ai_trading/tests/execution/test_*.py",
    "ai_trading/tests/optimization/test_*.py",
    "ai_trading/tests/ml/backtesting/test_*.py",
    "ai_trading/tests/ml/test_technical_indicators.py",
    "ai_trading/tests/ml/test_indicators.py",
    "ai_trading/tests/misc/test_complete_allocation_system.py",
    "ai_trading/tests/strategies/test_statistical_arbitrage.py",
)


def phase3_test_files() -> list[Path]:
    """Retourne une liste stable et dédupliquée des fichiers Phase 3."""
    files: set[Path] = set()
    for pattern in TEST_GROUPS:
        files.update(REPOSITORY_ROOT.glob(pattern))
    return sorted(files)


def main() -> int:
    failures: list[Path] = []
    test_files = phase3_test_files()
    current_process: list[subprocess.Popen[object] | None] = [None]

    def stop_current_test(_signum: int, _frame: object) -> None:
        """Évite qu'un pytest enfant survive à l'arrêt du lanceur."""
        process = current_process[0]
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop_current_test)
    signal.signal(signal.SIGINT, stop_current_test)
    if not test_files:
        raise RuntimeError("Aucun test Phase 3 trouvé")

    for index, test_file in enumerate(test_files, start=1):
        relative_path = test_file.relative_to(REPOSITORY_ROOT)
        print(f"\n[{index}/{len(test_files)}] {relative_path}", flush=True)
        current_process[0] = subprocess.Popen(
            [sys.executable, "-m", "pytest", str(relative_path), "-q", "-ra"],
            cwd=REPOSITORY_ROOT,
        )
        return_code = current_process[0].wait()
        current_process[0] = None
        if return_code not in {0, 5}:
            failures.append(relative_path)

    if failures:
        print("\nFichiers en échec :", *failures, sep="\n- ", file=sys.stderr)
        return 1
    print(f"\nPhase 3 validée : {len(test_files)} fichiers de test verts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
