import sys
from pathlib import Path
import numpy as np
import pandas as pd

# a projekt gyökér hozzáadása az importokhoz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig


def extract_features(df, config):
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    # idő és szesszió jellemzők
    # tört órával számolunk, mert a szesszió határok félórásak
    decimal_hours = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek + 1
    # hétvégén nincs kereskedés, ez csak a véletlen szombat/vasárnap gyertyákat fogja meg
    df["is_weekend"] = (df["day_of_week"] >= 6).astype(int)

    # óra és nap ciklikus kódolása sin/cos-szal
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)
    df["day_sin"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)

    is_weekday = df["is_weekend"] == 0

    df["is_asia"] = ((decimal_hours >= config.asia_start) & (decimal_hours < config.asia_end) & is_weekday).astype(int)
    df["is_london"] = ((decimal_hours >= config.london_start) & (decimal_hours < config.london_end) & is_weekday).astype(int)
    df["is_ny"] = ((decimal_hours >= config.ny_start) & (decimal_hours < config.ny_end) & is_weekday).astype(int)
    df["is_overlap"] = ((decimal_hours >= config.overlap_start) & (decimal_hours < config.overlap_end) & is_weekday).astype(int)

    # gyertya alak és több időtávú log hozamok
    eps = 1e-9
    df["body_pct"] = (df["close"] - df["open"]) / df["close"]
    candle_range = df["high"] - df["low"]
    # a kanócok aránya a teljes gyertyához képest
    df["wick_ratio"] = (candle_range - (df["close"] - df["open"]).abs()) / (candle_range + eps)

    for lag in config.return_lags:
        df[f"log_ret_{lag}"] = np.log(df["close"] / df["close"].shift(lag))

    # volatilitás (atr)
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # az alpha=1/periódus a wilder-féle simítással egyezik meg
    df["atr"] = tr.ewm(alpha=1.0 / config.atr_period, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    # ema távolság atr-rel normálva, tanh-val összenyomva
    ema_fast = df["close"].ewm(span=config.ema_fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=config.ema_slow_period, adjust=False).mean()

    scale_denom = config.tanh_dist_multiplier * df["atr"] + eps
    df["dist_ema50"] = np.tanh((df["close"] - ema_fast) / scale_denom)
    df["dist_ema200"] = np.tanh((df["close"] - ema_slow) / scale_denom)

    # rsi
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / config.rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / config.rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + eps)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # forgalom arány (tick volume)
    vol_sma = df["tick_volume"].rolling(window=config.vol_sma_period).mean()
    df["vol_ratio"] = df["tick_volume"] / (vol_sma + eps)

    # az indikátorok bemelegedési sorainak eldobása
    warmup_period = max(config.ema_slow_period, max(config.return_lags))
    return df.iloc[warmup_period:].reset_index(drop=True)


if __name__ == "__main__":
    cfg = PipelineConfig()

    input_path = PROJECT_ROOT / cfg.raw_data_path
    print(f"Adatok betöltése innen: {input_path}")

    df_raw = pd.read_csv(input_path)
    df_features = extract_features(df_raw, cfg)

    output_dir = PROJECT_ROOT / cfg.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"features_{cfg.symbol.lower()}_{cfg.timeframe.lower()}.csv"
    output_path = output_dir / output_filename

    df_features.to_csv(output_path, index=False)

    print("\n--- Feature engineering kész ---")
    print(f"Beolvasott sorok: {len(df_raw):,}")
    print(f"Kimeneti sorok: {len(df_features):,} (a bemelegedési sorok eldobása után)")
    print(f"Mentett fájl: {output_path}")
    print(f"Létrehozott jellemzők (összesen {len(df_features.columns)}):")
    print(list(df_features.columns))
