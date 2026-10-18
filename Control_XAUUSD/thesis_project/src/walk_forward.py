import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# a projekt gyökér hozzáadása
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.models import LSTMModel, XGBoostModel

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_us500_h1.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
PARAMS_FILE = REPORTS_DIR / "best_hyperparameters_us500.json"
OUTPUT_FILE = REPORTS_DIR / "walk_forward_results_us500.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_features_and_target(df):
    # ezeket az oszlopokat nem használjuk jellemzőnek
    excluded = {
        "time", "open", "high", "low", "close",
        "tick_volume", "spread", "atr", "hour", "day_of_week", "target"
    }
    return [c for c in df.columns if c not in excluded], "target"


def print_class_distribution_summary(y_true, xgb_preds, lstm_preds, tau, delta):
    total = len(y_true)
    labels = [-1, 0, 1]
    label_names = {-1: "Eladás (-1)", 0: "Tartás (0)", 1: "Vétel (+1)"}

    print("\n" + "=" * 80)
    print("US500 WALK-FORWARD ÖSSZESÍTETT OSZTÁLY-ELOSZLÁS ÉS PREDIKCIÓS ARÁNYOK")
    print(f"Alkalmazott döntési küszöbök: Tau (bizonyosság) >= {tau:.2f} | Delta (előny) > {delta:.2f}")
    print(f"Teljes Out-of-Sample minta: {total:,} óra".replace(",", " "))
    print("=" * 80)
    print(f"{'Kimenet / Osztály':<18} | {'Valós (True)':<16} | {'XGBoost':<16} | {'LSTM':<16}")
    print("-" * 80)

    for lbl in labels:
        name = label_names[lbl]
        c_tr = np.sum(y_true == lbl)
        c_xg = np.sum(xgb_preds == lbl)
        c_ls = np.sum(lstm_preds == lbl)
        print(
            f"{name:<18} | {c_tr:>6} ({c_tr/total*100:>5.1f}%)   | "
            f"{c_xg:>6} ({c_xg/total*100:>5.1f}%)   | "
            f"{c_ls:>6} ({c_ls/total*100:>5.1f}%)"
        )

    print("-" * 80)
    print(f"{'Összesen':<18} | {total:>6} (100.0%)   | {total:>6} (100.0%)   | {total:>6} (100.0%)")
    print("=" * 80)

    acc_xg = accuracy_score(y_true, xgb_preds)
    f1_xg = f1_score(y_true, xgb_preds, average="macro")
    acc_ls = accuracy_score(y_true, lstm_preds)
    f1_ls = f1_score(y_true, lstm_preds, average="macro")

    print(f"XGBoost | Out-of-Sample Macro-F1: {f1_xg:.4f} | Pontosság (Accuracy): {acc_xg:.4f}")
    print(f"LSTM    | Out-of-Sample Macro-F1: {f1_ls:.4f} | Pontosság (Accuracy): {acc_ls:.4f}")
    print("=" * 80 + "\n")


def run_walk_forward(n_splits=None, tau=0.42, delta=0.10):
    cfg = PipelineConfig()

    print("[*] US500 adatok betöltése a Walk-Forward vizsgálathoz...")
    df = pd.read_csv(DATA_FILE if DATA_FILE.exists() else PROJECT_ROOT / "labeled_us500_h1.csv")
    feature_cols, target_col = get_features_and_target(df)

    # hiperparaméterek betöltése
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Nem található az optimalizált paraméterfájl: {PARAMS_FILE}")

    print(f"[*] US500 hiperparaméterek beolvasása: {PARAMS_FILE}")
    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    xgb_params = best_params["xgboost"]["params"]
    lstm_params = best_params["lstm"]["params"].copy()
    lstm_params["input_dim"] = len(feature_cols)

    # időbeli szeletelés a config alapján
    total_bars = len(df)
    train_window_size = cfg.train_bars
    purging_gap = cfg.purge_bars
    test_fold_size = cfg.test_bars

    if total_bars <= train_window_size + test_fold_size:
        raise ValueError(
            f"Kevés az adat ({total_bars} sor) a {train_window_size} órás tanítóablakhoz "
            f"és a {test_fold_size} órás teszthez. Csökkentsd a train_bars-t a config.py-ban."
        )

    # ha nincs megadva, az ablakok száma az adat hosszából jön ki
    if n_splits is None:
        n_splits = max(1, (total_bars - train_window_size) // test_fold_size)

    all_fold_results = []
    print(f"[*] Indul a {n_splits} ablakos Rolling Walk-Forward US500 paraméterekkel...")

    extra_cols = [c for c in ["open", "high", "low", "spread"] if c in df.columns]

    for fold in range(n_splits):
        test_start = train_window_size + fold * test_fold_size
        test_end = (test_start + test_fold_size) if fold < n_splits - 1 else total_bars

        # mozgó tanítóablak
        train_start = max(0, test_start - train_window_size)
        train_slice = df.iloc[train_start : test_start - purging_gap]
        test_slice = df.iloc[test_start:test_end].copy()

        X_train, y_train = train_slice[feature_cols].values, train_slice[target_col].values
        X_test, y_test = test_slice[feature_cols].values, test_slice[target_col].values

        # xgboost
        xgb_clf = XGBoostModel(**xgb_params)
        xgb_clf.fit(X_train, y_train)
        xgb_probs = xgb_clf.predict_proba(X_test)
        xgb_preds = xgb_clf.predict(X_test, tau=tau, delta=delta)

        # lstm
        lstm_clf = LSTMModel(**lstm_params)
        lstm_clf.fit(X_train, y_train)

        # a teszt előtti valódi órák kellenek a szekvenciákhoz (csak jellemzők, címke nélkül)
        lookback = lstm_clf.sequence_length - 1
        pre_test = df[feature_cols].values[test_start - lookback : test_start]
        test_ext = np.vstack([pre_test, X_test])
        X_test_seq, _ = lstm_clf.create_sequences(test_ext)
        lstm_probs = lstm_clf.predict_proba(X_test_seq)
        lstm_preds = lstm_clf.predict(X_test_seq, tau=tau, delta=delta)

        # eredmények mentése (ohlc és spread is, hogy a backtest ne kérjen külön fájlt)
        fold_df = test_slice[["time", "close", "atr", target_col] + extra_cols].copy()
        fold_df["fold"] = fold + 1
        fold_df["xgb_pred"] = xgb_preds
        fold_df["xgb_prob_sell"] = xgb_probs[:, 0]
        fold_df["xgb_prob_hold"] = xgb_probs[:, 1]
        fold_df["xgb_prob_buy"] = xgb_probs[:, 2]

        fold_df["lstm_pred"] = lstm_preds
        fold_df["lstm_prob_sell"] = lstm_probs[:, 0]
        fold_df["lstm_prob_hold"] = lstm_probs[:, 1]
        fold_df["lstm_prob_buy"] = lstm_probs[:, 2]

        all_fold_results.append(fold_df)
        print(f"Fold {fold+1:02d}/{n_splits:02d} kész [{len(test_slice)} minta] | XGB F1: {f1_score(y_test, xgb_preds, average='macro'):.3f} | LSTM F1: {f1_score(y_test, lstm_preds, average='macro'):.3f}")

    # összes fold egyben
    results_df = pd.concat(all_fold_results, ignore_index=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[ok] Teljes US500 Walk-Forward eredménytábla elmentve: {OUTPUT_FILE}")

    print_class_distribution_summary(
        y_true=results_df[target_col].values.astype(int),
        xgb_preds=results_df["xgb_pred"].values.astype(int),
        lstm_preds=results_df["lstm_pred"].values.astype(int),
        tau=tau,
        delta=delta,
    )


if __name__ == "__main__":
    run_walk_forward(n_splits=None, tau=0.42, delta=0.10)
