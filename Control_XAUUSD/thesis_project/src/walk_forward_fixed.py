import sys
from pathlib import Path

# a projekt gyökér hozzáadása az importokhoz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import backtest, walk_forward

# rq2: ugyanaz a walk-forward és backtest, csak a fix horizontos címkékkel.
# a meglévő kódot használjuk, csak a be- és kimeneti fájlokat írjuk át,
# így az eredeti (triple barrier) eredmények nem íródnak felül
FIXED_DIR = PROJECT_ROOT / "reports" / "fixed_horizon"
FIXED_FIGURES_DIR = FIXED_DIR / "figures"
FIXED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # 1. walk-forward ugyanazokkal a hiperparaméterekkel, mint a triple barriernél
    walk_forward.DATA_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_fixed_us500_h1.csv"
    walk_forward.OUTPUT_FILE = FIXED_DIR / "walk_forward_results_us500.csv"
    walk_forward.run_walk_forward(n_splits=None, tau=0.42, delta=0.10)

    # 2. backtest ugyanazokkal a kereskedési szabályokkal
    backtest.WF_RESULTS_FILE = FIXED_DIR / "walk_forward_results_us500.csv"
    backtest.REPORTS_DIR = FIXED_DIR
    backtest.FIGURES_DIR = FIXED_FIGURES_DIR
    backtest.main()
