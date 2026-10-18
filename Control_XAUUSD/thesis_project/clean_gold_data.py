from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "data_us500_h1.csv"
BACKUP_FILE = PROJECT_ROOT / "data" / "raw" / "data_xauusd_h1_nyers.csv"


def main():
    df = pd.read_csv(RAW_FILE, parse_dates=["time"])
    df.to_csv(BACKUP_FILE, index=False)  # a tisztítatlan adat megmarad
    n_raw = len(df)

    # 1. a 00:00-s gyertyák 2021 tavaszáig napi összesítők (a záróáruk a nap végi ár),
    #    ezért jövőbeli információt tartalmaznak
    midnight = df["time"].dt.hour == 0
    df = df[~midnight].reset_index(drop=True)

    # 2. 2021 első felében a spread hibásan 0, ezt az adott óra
    #    nem nulla spreadjeinek mediánjával pótoljuk (csak a backtest költségét érinti)
    hour = df["time"].dt.hour
    hourly_median = df[df["spread"] > 0].groupby(hour[df["spread"] > 0])["spread"].median()
    zero_spread = df["spread"] == 0
    df.loc[zero_spread, "spread"] = hour[zero_spread].map(hourly_median).round().astype(int)

    df.to_csv(RAW_FILE, index=False)

    print(f"Beolvasott sorok: {n_raw:,}")
    print(f"Törölt 00:00-s gyertyák: {int(midnight.sum()):,}")
    print(f"Pótolt nulla spreadek: {int(zero_spread.sum()):,}")
    print(f"Megmaradt sorok: {len(df):,}")
    print(f"[ok] Tisztított adat mentve: {RAW_FILE}")
    print(f"[ok] Eredeti adat mentve: {BACKUP_FILE}")


if __name__ == "__main__":
    main()