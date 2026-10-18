from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WF_RESULTS_FILE = PROJECT_ROOT / "reports" / "walk_forward_results_us500.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

N_RUNS = 50
SEED = 42
CLASSES = np.array([-1, 0, 1])  # eladás, semleges, vétel


def random_baseline(y_true, n_runs=N_RUNS, seed=SEED):
    # egyenletes véletlen tippelés a három osztályon, n_runs futásban
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n_runs):
        y_pred = rng.choice(CLASSES, size=len(y_true))
        accs.append(accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return np.array(accs), np.array(f1s)


def main():
    df = pd.read_csv(WF_RESULTS_FILE)
    y_true = df["target"].values

    accs, f1s = random_baseline(y_true)

    summary = pd.DataFrame({
        "Viszonyítási alap": ["Véletlen tippelés (50 futás átlaga)"],
        "Pontosság (%)": [round(accs.mean() * 100, 2)],
        "Pontosság szórása (%)": [round(accs.std(ddof=1) * 100, 2)],
        "Macro-F1": [round(f1s.mean(), 3)],
        "Macro-F1 szórása": [round(f1s.std(ddof=1), 3)],
    })

    out = REPORTS_DIR / "random_baseline_us500.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"[ok] mentve: {out}")


if __name__ == "__main__":
    main()