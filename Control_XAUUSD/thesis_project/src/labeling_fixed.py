import sys
from pathlib import Path
import numpy as np
import pandas as pd

# a projekt gyökér hozzáadása az importokhoz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig


def apply_fixed_horizon(df, config, neutral_share):
    # fix horizontos címkézés az rq2-höz: csak a 24. gyertya záróára számít,
    # a köztes ármozgás nem, és a semleges sáv mindig ugyanakkora (nem függ az atr-től)
    df = df.copy()
    horizon = config.barrier_horizon
    n = len(df)

    close = df["close"].values
    future_return = np.full(n, np.nan)
    future_return[: n - horizon] = np.log(close[horizon:] / close[: n - horizon])

    # a végéről levágjuk, ahol nincs meg a teljes horizont (ugyanúgy, mint a triple barriernél)
    df = df.iloc[: n - horizon].copy().reset_index(drop=True)
    future_return = future_return[: n - horizon]

    # a semleges sáv szélességét csak az első tanítóablakból számoljuk,
    # hogy a tesztidőszakból ne kerüljön bele semmi
    first_window = np.abs(future_return[: config.train_bars])
    threshold = np.quantile(first_window, neutral_share)

    labels = np.zeros(len(df), dtype=int)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = -1
    df["target"] = labels

    return df, threshold


if __name__ == "__main__":
    cfg = PipelineConfig()
    data_dir = PROJECT_ROOT / cfg.processed_dir

    features_path = data_dir / f"features_{cfg.symbol.lower()}_{cfg.timeframe.lower()}.csv"
    tb_path = data_dir / f"labeled_{cfg.symbol.lower()}_{cfg.timeframe.lower()}.csv"
    output_path = data_dir / f"labeled_fixed_{cfg.symbol.lower()}_{cfg.timeframe.lower()}.csv"

    print(f"Jellemzők betöltése: {features_path}")
    df_features = pd.read_csv(features_path)
    df_tb = pd.read_csv(tb_path)

    # a semleges osztály aránya legyen ugyanannyi, mint a triple barriernél (első tanítóablak)
    tb_neutral_share = (df_tb["target"].values[: cfg.train_bars] == 0).mean()

    df_fixed, threshold = apply_fixed_horizon(df_features, cfg, tb_neutral_share)

    # ellenőrzés: ugyanazok a sorok legyenek, mint a triple barrier fájlban
    if len(df_fixed) != len(df_tb) or not (df_fixed["time"].values == df_tb["time"].values).all():
        raise ValueError("A fix horizontos és a triple barrier adatsor sorai nem egyeznek.")

    df_fixed.to_csv(output_path, index=False)

    print("\n--- Fix horizontos címkézés kész ---")
    print(f"Horizont: {cfg.barrier_horizon} gyertya")
    print(f"Semleges sáv: |hozam| <= {threshold * 100:.3f}%")
    print(f"Mentett fájl: {output_path}")
    print(f"Egyezés a triple barrier címkékkel: {(df_fixed['target'] == df_tb['target']).mean() * 100:.2f}%")

    # osztályeloszlás kiírása a kettő mellett
    print("\nCímke eloszlás:        Fix horizont   Triple barrier")
    for label_val, name in [(1, "Long"), (-1, "Short"), (0, "Semleges")]:
        fixed_pct = (df_fixed["target"] == label_val).mean() * 100
        tb_pct = (df_tb["target"] == label_val).mean() * 100
        print(f"  Osztály {label_val:>2} [{name:<8}]   {fixed_pct:6.2f}%        {tb_pct:6.2f}%")
