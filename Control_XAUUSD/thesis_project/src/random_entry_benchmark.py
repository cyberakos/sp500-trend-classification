import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtest import calculate_metrics, simulate_trades  # noqa: E402

WF_RESULTS_FILE = PROJECT_ROOT / "reports" / "walk_forward_results_us500.csv"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "random_entry_benchmark_us500.csv"

N_RUNS = 200
SEED = 42
INITIAL_CAPITAL = 1000.0

# ugyanazok a szabályok, mint a backtest.py-ban
SIM_KWARGS = dict(
    initial_capital=INITIAL_CAPITAL,
    risk_pct=0.010,
    tau=0.52,
    delta=0.10,
    sl_mult=2.0,
    pt_mult=2.5,
    max_holding_bars=24,
    cooldown_bars=12,
    slippage_usd=0.15,
    default_spread_usd=0.40,
)

METRICS = ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Profit Factor", "Total Trades"]


def model_signal_rate(df, model):
    # a modell az órák mekkora részén teljesíti a belépési szabályt
    p_buy = df[f"{model}_prob_buy"]
    p_sell = df[f"{model}_prob_sell"]
    return ((p_buy >= SIM_KWARGS["tau"]) & ((p_buy - p_sell) > SIM_KWARGS["delta"])).mean()


def mean_trade_count(df, signal_rate, rng, n_runs=10):
    counts = []
    for _ in range(n_runs):
        sim = df.copy()
        signal = rng.random(len(sim)) < signal_rate
        sim["rnd_prob_buy"] = np.where(signal, 1.0, 0.0)
        sim["rnd_prob_sell"] = 0.0
        _, trades = simulate_trades(sim, model_name="rnd", **SIM_KWARGS)
        counts.append(len(trades))
    return np.mean(counts)


def calibrate_rate(df, target_trades, rng, upper):
    # a modell jelzései időben csomósodnak, ezért azonos jelzésgyakoriság mellett
    # a véletlen stratégia több kötést kötne; felezéssel keressük meg azt a
    # gyakoriságot, amelynél a kötések átlagos száma megegyezik a modellével
    lo, hi = 0.0005, upper
    for _ in range(12):
        mid = (lo + hi) / 2
        if mean_trade_count(df, mid, rng) < target_trades:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def random_runs(df, signal_rate, rng):
    # véletlen belépési jelek ugyanakkora gyakorisággal, mint a modellé
    rows = []
    for _ in range(N_RUNS):
        sim = df.copy()
        signal = rng.random(len(sim)) < signal_rate
        sim["rnd_prob_buy"] = np.where(signal, 1.0, 0.0)
        sim["rnd_prob_sell"] = 0.0
        equity, trades = simulate_trades(sim, model_name="rnd", **SIM_KWARGS)
        rows.append(calculate_metrics(equity, trades, INITIAL_CAPITAL))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(WF_RESULTS_FILE)
    df["time"] = pd.to_datetime(df["time"])
    rng = np.random.default_rng(SEED)

    summary = []
    for model, name in [("xgb", "XGBoost"), ("lstm", "LSTM")]:
        equity, trades = simulate_trades(df, model_name=model, **SIM_KWARGS)
        model_metrics = calculate_metrics(equity, trades, INITIAL_CAPITAL)

        model_rate = model_signal_rate(df, model)
        rate = calibrate_rate(df, len(trades), rng, upper=model_rate)
        rnd = random_runs(df, rate, rng)

        for metric in METRICS:
            values = rnd[metric]
            summary.append({
                "Modell": name,
                "Modell jelzésgyakorisága (%)": round(model_rate * 100, 2),
                "Véletlen jelzésgyakoriság (%)": round(rate * 100, 2),
                "Mutató": metric,
                "Modell értéke": model_metrics[metric],
                "Véletlen átlag": round(values.mean(), 2),
                "Véletlen 5%": round(values.quantile(0.05), 2),
                "Véletlen 95%": round(values.quantile(0.95), 2),
                # a véletlen futások hány százaléka rosszabb a modellnél
                "Modell percentilise (%)": round((values < model_metrics[metric]).mean() * 100, 1),
            })

    out = pd.DataFrame(summary)
    out.to_csv(OUTPUT_FILE, index=False)
    print(out.to_string(index=False))
    print(f"\n[ok] mentve: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()