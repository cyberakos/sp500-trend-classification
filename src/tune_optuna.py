import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.metrics import f1_score

# a projekt gyökér hozzáadása az importokhoz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.models import LSTMModel, XGBoostModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_us500_h1.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_features_and_target(df):
    # ezeket az oszlopokat nem használjuk jellemzőnek
    excluded = {
        "time", "open", "high", "low", "close",
        "tick_volume", "spread", "atr", "hour", "day_of_week", "target"
    }
    return [c for c in df.columns if c not in excluded], "target"


def main():
    print("[*] US500 adatok betöltése az Optuna hangoláshoz...")
    df = pd.read_csv(DATA_FILE if DATA_FILE.exists() else PROJECT_ROOT / "labeled_us500_h1.csv")
    feature_cols, target_col = get_features_and_target(df)

    # csak az első walk-forward teszt előtti adaton hangolunk,
    # így az optuna nem lát rá az out-of-sample részre
    cfg = PipelineConfig()
    tune_end = cfg.train_bars - cfg.purge_bars
    tune_df = df.iloc[:tune_end]

    # ezen belül a train 80%, a val 20%, köztük 24 órás kihagyással
    n = len(tune_df)
    train_end = int(n * 0.80)

    train_df = tune_df.iloc[: train_end - 24]
    val_df = tune_df.iloc[train_end:]
    print(f"[*] hangolás az első {n} báron (train: {len(train_df)}, val: {len(val_df)})")

    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_val, y_val = val_df[feature_cols].values, val_df[target_col].values

    best_results = {}

    # xgboost hangolás (30 kísérlet)
    print("\n[*] 1. XGBoost optimalizálás US500-ra (30 kísérlet)...")

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9, step=0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 8.0, log=True),
        }
        model = XGBoostModel(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="macro")

    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=True)
    best_results["xgboost"] = {"best_macro_f1": round(xgb_study.best_value, 4), "params": xgb_study.best_params}
    print(f"[ok] Legjobb US500 XGBoost Macro-F1: {xgb_study.best_value:.4f}")

    # lstm hangolás (12 kísérlet, gpu-n)
    print("\n[*] 2. LSTM optimalizálás US500-ra GPU-n (12 kísérlet, gyorsított batch-ekkel)...")

    def lstm_objective(trial):
        params = {
            "input_dim": len(feature_cols),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [48, 64, 96]),
            "dropout": trial.suggest_float("dropout", 0.20, 0.40, step=0.05),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
            "epochs": 15,
            "patience": 3,
        }
        model = LSTMModel(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        X_val_seq, y_val_seq = model.create_sequences(X_val, y_val)
        preds = model.predict(X_val_seq)
        return f1_score(y_val_seq, preds, average="macro")

    lstm_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    lstm_study.optimize(lstm_objective, n_trials=12, show_progress_bar=True)
    best_results["lstm"] = {"best_macro_f1": round(lstm_study.best_value, 4), "params": lstm_study.best_params}
    print(f"[ok] Legjobb US500 LSTM Macro-F1: {lstm_study.best_value:.4f}")

    # mentés
    json_path = REPORTS_DIR / "best_hyperparameters_us500.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=4)
    print(f"\n[ok] US500 hiperparaméterek elmentve: {json_path}")

    # paraméter fontosság ábrák
    for study, name in [(xgb_study, "xgb"), (lstm_study, "lstm")]:
        try:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title(f"{name.upper()} Paraméter-érzékenység (US500)", fontweight="bold")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"optuna_{name}_importance_us500.png", dpi=300)
            plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()