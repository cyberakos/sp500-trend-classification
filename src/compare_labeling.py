import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

# a projekt gyökér hozzáadása az importokhoz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import simulate_trades

REPORTS_DIR = PROJECT_ROOT / "reports"
FIXED_DIR = REPORTS_DIR / "fixed_horizon"
TB_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_us500_h1.csv"

# a két futás walk-forward és backtest eredményei
WF_FILES = {
    "Triple Barrier": REPORTS_DIR / "walk_forward_results_us500.csv",
    "Fix horizont": FIXED_DIR / "walk_forward_results_us500.csv",
}
BACKTEST_FILES = {
    "Triple Barrier": REPORTS_DIR / "backtest_metrics_comparison_us500.csv",
    "Fix horizont": FIXED_DIR / "backtest_metrics_comparison_us500.csv",
}
MODELS = {"xgb": "XGBoost", "lstm": "LSTM"}

# a backtest belépési szabálya (backtest.py)
ENTRY_TAU = 0.52
ENTRY_DELTA = 0.10


def load_results():
    # a triple barrier címkét mindkét táblához hozzátesszük, ez lesz a közös mérce
    tb_labels = pd.read_csv(TB_DATA_FILE, usecols=["time", "target"])
    tb_labels = tb_labels.rename(columns={"target": "tb_target"})

    results = {}
    for name, path in WF_FILES.items():
        df = pd.read_csv(path)
        results[name] = df.merge(tb_labels, on="time", how="left")
    return results


def classification_table(results):
    # mindkét címkézésnél a saját címkéihez mérjük a modellt (argmax predikcióval)
    rows = []
    for labeling, df in results.items():
        y_true = df["target"].values

        # naiv alap: mindig long
        always_long = np.ones(len(y_true), dtype=int)
        rows.append({
            "Címkézés": labeling,
            "Modell": "Mindig Long",
            "Pontosság": accuracy_score(y_true, always_long),
            "Macro-F1": f1_score(y_true, always_long, average="macro", zero_division=0),
            "Cohen-kappa": 0.0,
        })

        for model, model_name in MODELS.items():
            probs = df[[f"{model}_prob_sell", f"{model}_prob_hold", f"{model}_prob_buy"]].values
            y_pred = np.argmax(probs, axis=1) - 1
            rows.append({
                "Címkézés": labeling,
                "Modell": model_name,
                "Pontosság": accuracy_score(y_true, y_pred),
                "Macro-F1": f1_score(y_true, y_pred, average="macro"),
                "Cohen-kappa": cohen_kappa_score(y_true, y_pred),
            })
    return pd.DataFrame(rows).round(4)


def signal_quality_table(results):
    # közös mérce: a backtest vételi jeleinél hányszor éri el az ár előbb a felső (+2.5 atr)
    # korlátot, mint az alsót. ez mindkét címkézésnél ugyanúgy mérhető
    rows = []
    hit_rate_per_fold = {}
    for model, model_name in MODELS.items():
        for labeling, df in results.items():
            p_buy = df[f"{model}_prob_buy"].values
            p_sell = df[f"{model}_prob_sell"].values
            signal = (p_buy >= ENTRY_TAU) & ((p_buy - p_sell) > ENTRY_DELTA)
            outcome = df.loc[signal, "tb_target"]

            rows.append({
                "Modell": model_name,
                "Címkézés": labeling,
                "Vételi jelek": int(signal.sum()),
                "Felső korlát előbb (%)": (outcome == 1).mean() * 100,
                "Alsó korlát előbb (%)": (outcome == -1).mean() * 100,
                "Lejárt az idő (%)": (outcome == 0).mean() * 100,
                "Alapráta: felső korlát az összes órán (%)": (df["tb_target"] == 1).mean() * 100,
            })

            # ablakonként is kiszámoljuk a páros próbához
            signals = df[signal]
            hit_rate_per_fold[(model, labeling)] = (signals["tb_target"] == 1).groupby(signals["fold"]).mean()

    # páros wilcoxon próba a 29 ablakon (triple barrier vs fix horizont)
    test_rows = []
    for model, model_name in MODELS.items():
        tb = hit_rate_per_fold[(model, "Triple Barrier")]
        fixed = hit_rate_per_fold[(model, "Fix horizont")]
        folds = tb.index.intersection(fixed.index)
        diff = tb[folds] - fixed[folds]
        _, p_value = wilcoxon(tb[folds], fixed[folds])
        test_rows.append({
            "Modell": model_name,
            "Ablakok": len(folds),
            "TB jobb ablakban": int((diff > 0).sum()),
            "Átlagos különbség TB - FH (százalékpont)": diff.mean() * 100,
            "Wilcoxon p-érték": p_value,
        })

    return pd.DataFrame(rows).round(2), pd.DataFrame(test_rows).round(4)


def backtest_table():
    # a két backtest metrikái egymás mellett
    table = pd.DataFrame()
    for labeling, path in BACKTEST_FILES.items():
        df = pd.read_csv(path).set_index("Mutató (Metrika)")
        table[f"XGBoost – {labeling}"] = df["XGBoost US500"]
        table[f"LSTM – {labeling}"] = df["LSTM US500"]
    table["Buy & Hold"] = df["Buy & Hold US500"]
    return table


def plot_equity_curves(results):
    # a tőkegörbéket újra lefuttatjuk ugyanazzal a szimulátorral, hogy egy ábrán legyenek
    sim_kwargs = dict(
        initial_capital=1000.0, risk_pct=0.010, tau=ENTRY_TAU, delta=ENTRY_DELTA,
        sl_mult=2.0, pt_mult=2.5, max_holding_bars=24, cooldown_bars=12,
        slippage_usd=0.15, default_spread_usd=0.40,
    )
    colors = {"xgb": "#1f77b4", "lstm": "#ff7f0e"}
    line_styles = {"Triple Barrier": "-", "Fix horizont": ":"}

    plt.figure(figsize=(12, 6), dpi=300)
    for model, model_name in MODELS.items():
        for labeling, df in results.items():
            equity, _ = simulate_trades(df, model_name=model, **sim_kwargs)
            total_return = (equity.iloc[-1] / 1000.0 - 1) * 100
            plt.plot(equity.index, equity.values, color=colors[model], linestyle=line_styles[labeling],
                     linewidth=1.6, label=f"{model_name} – {labeling} ({total_return:+.1f}%)")

    df = results["Triple Barrier"]
    buy_hold = 1000.0 * df["close"] / df["close"].iloc[0]
    plt.plot(pd.to_datetime(df["time"]), buy_hold.values, color="#7f7f7f", linestyle="--", alpha=0.6,
             label=f"Buy & Hold ({(buy_hold.iloc[-1] / 1000.0 - 1) * 100:+.1f}%)")

    plt.title("Tőkegörbék: Triple Barrier vs. fix horizontos címkézés - US500 H1", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Dátum", fontsize=11)
    plt.ylabel("Tőke (USD)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIXED_DIR / "figures" / "equity_tb_vs_fixed_us500.png")
    plt.close()


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    results = load_results()

    print("\n" + "=" * 90)
    print("1. OSZTÁLYOZÁS (mindkettő a saját címkéihez mérve, argmax predikció)")
    print("=" * 90)
    classification = classification_table(results)
    print(classification.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"2. KÖZÖS MÉRCE: a vételi jelek kimenete a ±2,5 ATR korlátokon (p_buy >= {ENTRY_TAU}, előny > {ENTRY_DELTA})")
    print("=" * 90)
    signals, tests = signal_quality_table(results)
    print(signals.to_string(index=False))
    print("\nPáros Wilcoxon-próba ablakonként (felső korlát előbb arány):")
    print(tests.to_string(index=False))

    print("\n" + "=" * 90)
    print("3. BACKTEST (ugyanazok a kereskedési szabályok, csak a tanító címke más)")
    print("=" * 90)
    backtest_metrics = backtest_table()
    print(backtest_metrics.to_string())

    plot_equity_curves(results)

    # mentés
    classification.to_csv(FIXED_DIR / "rq2_classification_us500.csv", index=False)
    signals.to_csv(FIXED_DIR / "rq2_signal_quality_us500.csv", index=False)
    tests.to_csv(FIXED_DIR / "rq2_wilcoxon_us500.csv", index=False)
    backtest_metrics.to_csv(FIXED_DIR / "rq2_backtest_comparison_us500.csv")
    print(f"\n[ok] RQ2 táblák és ábra elmentve ide: {FIXED_DIR}")
