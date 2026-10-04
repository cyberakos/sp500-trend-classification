import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

# a projekt gyökér hozzáadása
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import LSTMModel, XGBoostModel

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_us500_h1.csv"
PARAMS_FILE = PROJECT_ROOT / "reports" / "best_hyperparameters_us500.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_features_and_target(df):
    # ezeket az oszlopokat nem használjuk jellemzőnek
    excluded = {
        "time", "open", "high", "low", "close",
        "tick_volume", "spread", "atr", "hour", "day_of_week", "target"
    }
    return [c for c in df.columns if c not in excluded], "target"


def main():
    print("[*] US500 adatok betöltése...")
    df = pd.read_csv(DATA_FILE if DATA_FILE.exists() else PROJECT_ROOT / "labeled_us500_h1.csv")
    feature_cols, target_col = get_features_and_target(df)

    # a train 70%, val 15%, test 15%, köztük 24 órás kihagyással
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[: train_end - 24]
    val_df = df.iloc[train_end : val_end - 24]
    test_df = df.iloc[val_end:].copy()

    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
    X_test, y_test = test_df[feature_cols].values, test_df[target_col].values

    # küszöbök a zaj szűréséhez
    tau = 0.42
    delta = 0.10

    # hiperparaméterek betöltése
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Nem található: {PARAMS_FILE} (előbb futtasd a tune_optuna.py-t)")
    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        best = json.load(f)

    # xgboost tanítása
    print("[*] XGBoost tanítása az US500 paraméterekkel...")
    xgb_model = XGBoostModel(**best["xgboost"]["params"])
    xgb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    xgb_model.save(MODELS_DIR / "xgboost_single_split_us500.json")
    xgb_preds = xgb_model.predict(X_test, tau=tau, delta=delta)

    # lstm tanítása
    print("[*] LSTM tanítása GPU-n az US500 paraméterekkel...")
    lstm_params = best["lstm"]["params"].copy()
    lstm_params["input_dim"] = len(feature_cols)  # az optuna ezt nem menti el, de kell a modellnek
    lstm_model = LSTMModel(**lstm_params)
    lstm_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    lstm_model.save(MODELS_DIR / "lstm_single_split_us500.pt")

    # teszt szekvenciák: elé tesszük a val utolsó óráit
    test_ext = np.vstack([val_df[feature_cols].values[-(lstm_model.sequence_length - 1):], X_test])
    X_test_seq, _ = lstm_model.create_sequences(test_ext)
    probs = lstm_model.predict_proba(X_test_seq)
    print("max valószínűségek átlaga:", probs.max(axis=1).mean())
    print("p_buy min/max:", probs[:, 2].min(), probs[:, 2].max())
    print("argmax eloszlás:", np.bincount(np.argmax(probs, axis=1), minlength=3))
    for name, probs in [("xgb", xgb_model.predict_proba(X_test)), ("lstm", lstm_model.predict_proba(X_test_seq))]:
        p_sell, p_buy = probs[:, 0], probs[:, 2]
        gap = p_buy - p_sell
        print(name, "p_buy>=tau:", (p_buy >= tau).sum(), "| gap>delta:", (gap > delta).sum(),
              "| mindkettő:", ((p_buy >= tau) & (gap > delta)).sum())
        print(name, "gap percentilisek:", np.percentile(np.abs(gap), [50, 90, 99]))
    lstm_preds = lstm_model.predict(X_test_seq, tau=tau, delta=delta)

    # kiértékelés
    print("=" * 65)
    print("EGYSZERI STATIKUS FELOSZTÁS EREDMÉNYEI (US500 H1)")
    print("=" * 65)
    print(f"XGBoost | Acc: {accuracy_score(y_test, xgb_preds):.4f} | Macro-F1: {f1_score(y_test, xgb_preds, average='macro'):.4f}")
    print(f"LSTM    | Acc: {accuracy_score(y_test, lstm_preds):.4f} | Macro-F1: {f1_score(y_test, lstm_preds, average='macro'):.4f}")
    print("=" * 65)

    # ábrák mentése
    save_learning_curves(xgb_model, lstm_model)
    save_confusion_matrices(y_test, xgb_preds, lstm_preds)
    print("[ok] US500 modellek és diagramok sikeresen elmentve!")


def save_learning_curves(xgb_model, lstm_model):
    evals = xgb_model.model.evals_result()
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(evals["validation_0"]["mlogloss"], label="Train Loss", color="#1f77b4")
    plt.plot(evals["validation_1"]["mlogloss"], label="Val Loss", color="#d62728")
    plt.title("XGBoost Tanulási Görbe - US500", fontweight="bold")
    plt.xlabel("Iteráció (Fák száma)")
    plt.ylabel("Multi-class Log-Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgb_loss_curve_us500.png")
    plt.close()

    hist = lstm_model.history_
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    ax1.plot(hist["train_loss"], label="Train Loss", color="#1f77b4")
    ax1.plot(hist["val_loss"], label="Val Loss", color="#d62728")
    ax1.set_title("LSTM Keresztentrópia Veszteség - US500", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(hist["train_acc"], label="Train Acc", color="#2ca02c")
    ax2.plot(hist["val_acc"], label="Val Acc", color="#ff7f0e")
    ax2.set_title("LSTM Pontosság (Accuracy) - US500", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lstm_learning_curves_us500.png")
    plt.close()


def save_confusion_matrices(y_true, xgb_pred, lstm_pred):
    labels = ["Eladás (-1)", "Tartás (0)", "Vétel (+1)"]
    class_values = [-1, 0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    cm_xgb = confusion_matrix(y_true, xgb_pred, labels=class_values, normalize="true")
    ConfusionMatrixDisplay(cm_xgb, display_labels=labels).plot(ax=axes[0], cmap="Blues", values_format=".2f", colorbar=False)
    axes[0].set_title("XGBoost Konfúziós Mátrix - US500", fontweight="bold")

    cm_lstm = confusion_matrix(y_true, lstm_pred, labels=class_values, normalize="true")
    ConfusionMatrixDisplay(cm_lstm, display_labels=labels).plot(ax=axes[1], cmap="Oranges", values_format=".2f", colorbar=False)
    axes[1].set_title("LSTM Konfúziós Mátrix - US500", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices_single_us500.png")
    plt.close()


if __name__ == "__main__":
    main()
