import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# a projekt gyökér hozzáadása az importokhoz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import LSTMModel, XGBoostModel

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_us500_h1.csv"
PARAMS_FILE = REPORTS_DIR / "best_hyperparameters_us500.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_features_and_target(df):
    # ezeket az oszlopokat nem használjuk jellemzőnek
    excluded = {
        "time", "open", "high", "low", "close",
        "tick_volume", "spread", "atr", "hour", "day_of_week", "target"
    }
    return [c for c in df.columns if c not in excluded], "target"


def get_feature_category(col):
    if "log_ret" in col: return "Log Hozamok (Lags)"
    if "dist_ema" in col or "atr_pct" in col: return "Volatilitás & Trend"
    if col in ["rsi", "vol_ratio"]: return "Momentum & Forgalom"
    if "is_" in col and col != "is_weekend": return "Kereskedési Szakaszok"
    if "hour" in col or "day" in col or col == "is_weekend": return "Ciklikus Idő"
    return "Gyertyamorfológia"


def plot_importance_bar(df, value_col, title, color, output_path):
    # vízszintes oszlopdiagram, fentről lefelé csökkenő sorrendben
    sorted_df = df.sort_values(value_col, ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 7), dpi=300)
    bars = plt.barh(sorted_df["Feature"], sorted_df[value_col], color=color, edgecolor="black", alpha=0.85)
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relatív fontosság (%)", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    max_val = max(sorted_df[value_col].max(), 5.0)
    plt.xlim(0, max_val * 1.15)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + (max_val * 0.015), bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[ok] Ábra elmentve: {output_path}")


def main():
    print("[*] US500 adatok betöltése...")
    df = pd.read_csv(DATA_FILE if DATA_FILE.exists() else PROJECT_ROOT / "labeled_us500_h1.csv")
    feature_cols, target_col = get_features_and_target(df)

    # ugyanaz a felosztás, mint a train_single_split.py-ban
    total_bars = len(df)
    train_end = int(total_bars * 0.70)
    val_end = int(total_bars * 0.85)
    val_df = df.iloc[train_end : val_end - 24].copy()
    test_df = df.iloc[val_end:].copy()

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values.astype(int)

    # a mentett modellek ezekkel a paraméterekkel készültek
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Nem található: {PARAMS_FILE}")
    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        best = json.load(f)

    # xgboost fontosság (gain)
    print("[*] 1. XGBoost modell betöltése és fontossági súlyok számítása...")
    xgb_model = XGBoostModel(**best["xgboost"]["params"]).load(MODELS_DIR / "xgboost_single_split_us500.json")
    xgb_raw_imp = xgb_model.model.feature_importances_
    xgb_imp_pct = (xgb_raw_imp / np.sum(xgb_raw_imp)) * 100.0

    # lstm fontosság (permutációs)
    print("[*] 2. LSTM modell betöltése és permutációs kísérletek futtatása...")
    lstm_params = best["lstm"]["params"].copy()
    lstm_params["input_dim"] = len(feature_cols)
    lstm_model = LSTMModel(**lstm_params).load(MODELS_DIR / "lstm_single_split_us500.pt")

    # a teszt elé betesszük a val utolsó óráit a szekvenciákhoz
    X_test_ext = np.vstack([val_df[feature_cols].values[-(lstm_model.sequence_length - 1):], X_test])
    X_ts_seq, _ = lstm_model.create_sequences(X_test_ext)

    base_preds = lstm_model.predict(X_ts_seq)
    baseline_f1 = f1_score(y_test, base_preds, average="macro")
    print(f"[*] LSTM Teszt Alapértelmezett Macro-F1: {baseline_f1:.4f}")

    lstm_drops = []
    rng = np.random.default_rng(42)

    for i in range(len(feature_cols)):
        X_perm = X_test_ext.copy()
        rng.shuffle(X_perm[:, i])  # csak az i-edik oszlopot keverjük össze

        X_perm_seq, _ = lstm_model.create_sequences(X_perm)
        perm_preds = lstm_model.predict(X_perm_seq)
        drop = max(0.0, baseline_f1 - f1_score(y_test, perm_preds, average="macro"))
        lstm_drops.append(drop)

    lstm_drops = np.array(lstm_drops)
    if np.sum(lstm_drops) > 0:
        lstm_imp_pct = lstm_drops / np.sum(lstm_drops) * 100.0
    else:
        lstm_imp_pct = np.full(len(feature_cols), 100.0 / len(feature_cols))

    # táblázat és diagramok mentése
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "XGBoost_Importance_Pct": np.round(xgb_imp_pct, 2),
        "LSTM_Importance_Pct": np.round(lstm_imp_pct, 2),
    })
    importance_df["Category"] = importance_df["Feature"].apply(get_feature_category)

    csv_path = REPORTS_DIR / "feature_importance_comparison_us500.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"[ok] Összesítő táblázat elmentve: {csv_path}")

    plot_importance_bar(
        importance_df, "XGBoost_Importance_Pct",
        "XGBoost Jellemzőfontosság - US500 (MDI Gain %)", "#1f77b4",
        FIGURES_DIR / "xgb_feature_importance_us500.png"
    )
    plot_importance_bar(
        importance_df, "LSTM_Importance_Pct",
        "LSTM Jellemzőfontosság - US500 (Permutációs MDA %)", "#ff7f0e",
        FIGURES_DIR / "lstm_feature_importance_us500.png"
    )


if __name__ == "__main__":
    main()
