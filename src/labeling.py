import sys
from pathlib import Path
import numpy as np
import pandas as pd

# a projekt gyökér hozzáadása az importokhoz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig


def apply_triple_barrier(df, config):
    df = df.copy()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    n = len(df)

    horizon = config.barrier_horizon
    multiplier = config.barrier_multiplier

    labels = np.zeros(n, dtype=int)

    # minden bárnál előre megyünk a horizont végéig
    for i in range(n - horizon):
        entry_price = close[i]
        barrier_distance = atr[i] * multiplier

        upper_barrier = entry_price + barrier_distance
        lower_barrier = entry_price - barrier_distance

        future_highs = high[i + 1 : i + 1 + horizon]
        future_lows = low[i + 1 : i + 1 + horizon]

        upper_hits = np.where(future_highs >= upper_barrier)[0]
        lower_hits = np.where(future_lows <= lower_barrier)[0]

        first_upper = upper_hits[0] if len(upper_hits) > 0 else np.inf
        first_lower = lower_hits[0] if len(lower_hits) > 0 else np.inf

        # amelyik sáv előbb törik át, az adja a címkét
        # egyszerre törés vagy nincs törés esetén marad a 0
        if first_upper < first_lower:
            labels[i] = 1
        elif first_lower < first_upper:
            labels[i] = -1

    df["target"] = labels

    # a végéről levágjuk, ahol nincs meg a teljes horizont
    valid_df = df.iloc[: n - horizon].copy().reset_index(drop=True)
    return valid_df


if __name__ == "__main__":
    cfg = PipelineConfig()

    input_dir = PROJECT_ROOT / cfg.processed_dir
    input_filename = f"features_{cfg.symbol.lower()}_{cfg.timeframe.lower()}.csv"
    input_path = input_dir / input_filename

    print(f"Jellemzők betöltése: {input_path}")
    df_features = pd.read_csv(input_path)

    df_labeled = apply_triple_barrier(df_features, cfg)

    output_filename = f"labeled_{cfg.symbol.lower()}_{cfg.timeframe.lower()}.csv"
    output_path = input_dir / output_filename
    df_labeled.to_csv(output_path, index=False)

    print("\n--- Triple barrier címkézés kész ---")
    print(f"Címkézett sorok száma: {len(df_labeled):,}")
    print(f"Mentett fájl: {output_path}")

    # osztályeloszlás kiírása
    target_counts = df_labeled["target"].value_counts().sort_index()
    target_pcts = df_labeled["target"].value_counts(normalize=True).sort_index() * 100

    print("\nCímke eloszlás:")
    for label_val, count in target_counts.items():
        pct = target_pcts[label_val]
        name = {1: "Long (felső sáv)", -1: "Short (alsó sáv)", 0: "Semleges (lejárt az idő)"}.get(label_val)
        print(f"  Osztály {label_val:>2} [{name}]: {count:>6,} bár ({pct:5.2f}%)")
