from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_us500_h1.csv"
WF_RESULTS_FILE = PROJECT_ROOT / "reports" / "walk_forward_results_us500.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# az mt5 spread oszlopa pontban van, us500-nál 1 pont = 0.01 indexpont (40 pont = 0.40)
SPREAD_POINT_SIZE = 0.01


def simulate_trades(
    df,
    model_name,
    initial_capital=10000.0,
    risk_pct=0.01,          # pozíciónként 1% kockázat
    tau=0.46,               # belépési bizonyosság
    delta=0.08,             # irány előny
    sl_mult=2.0,            # stop-loss atr szorzó
    pt_mult=2.5,            # take-profit atr szorzó
    max_holding_bars=24,    # maximum 24 órát tartunk
    cooldown_bars=12,       # zárás után ennyi óra várakozás
    slippage_usd=0.15,      # slippage indexpontban
    default_spread_usd=0.40,  # spread indexpontban, ha nincs az adatban
):
    # long-only szimuláció, csak az ml valószínűségek alapján
    equity = initial_capital
    bar_equity = np.zeros(len(df))
    trade_records = []

    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    units = 0.0
    last_exit_idx = -cooldown_bars

    times = df["time"].values
    opens = df["open"].values if "open" in df.columns else df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr"].values
    if "spread" in df.columns:
        spreads = df["spread"].values * SPREAD_POINT_SIZE
    else:
        spreads = np.full(len(df), default_spread_usd)

    p_buys = df[f"{model_name}_prob_buy"].values
    p_sells = df[f"{model_name}_prob_sell"].values

    for i in range(len(df)):
        c_time = times[i]
        c_open = opens[i]
        c_close = closes[i]
        c_high = highs[i]
        c_low = lows[i]
        c_atr = atrs[i]
        c_spread = spreads[i]

        # nyitott pozíció kezelése
        if in_trade:
            bars_held = i - entry_idx
            exit_trade = False
            exit_price = 0.0
            exit_reason = ""

            # az indexek gyakran gap-elnek (napi szünet, hétfő nyitás),
            # ilyenkor a nyitóár a kilépő ár, nem az sl szint
            if c_low <= sl_price:
                exit_trade = True
                exit_price = min(sl_price, c_open) - slippage_usd
                exit_reason = "SL"
            elif c_high >= tp_price:
                exit_trade = True
                exit_price = max(tp_price, c_open) - slippage_usd
                exit_reason = "TP"
            elif bars_held >= max_holding_bars:
                exit_trade = True
                exit_price = c_close - slippage_usd - (c_spread / 2.0)
                exit_reason = "TIME"

            if exit_trade:
                pnl = (exit_price - entry_price) * units
                equity += pnl
                in_trade = False
                last_exit_idx = i
                trade_records.append({
                    "entry_time": times[entry_idx],
                    "exit_time": c_time,
                    "type": "BUY",
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "return_pct": round(pnl / (equity - pnl) * 100, 3),
                    "reason": exit_reason,
                    "bars_held": bars_held,
                    "equity_after": round(equity, 2),
                })

        # új belépés: csak az ml valószínűségek alapján
        if not in_trade and (i - last_exit_idx >= cooldown_bars) and equity > 0:
            pb, ps = p_buys[i], p_sells[i]

            if (pb >= tau) and ((pb - ps) > delta):
                entry_idx = i
                risk_amount = equity * risk_pct
                sl_distance = sl_mult * c_atr
                if sl_distance <= 0:
                    sl_distance = c_close * 0.01

                units = risk_amount / sl_distance
                entry_price = c_close + slippage_usd + (c_spread / 2.0)
                sl_price = entry_price - sl_distance
                tp_price = entry_price + (pt_mult * c_atr)
                in_trade = True

        bar_equity[i] = equity

    equity_series = pd.Series(bar_equity, index=pd.to_datetime(df["time"]))
    trades_df = pd.DataFrame(trade_records)
    return equity_series, trades_df


def calculate_metrics(equity_series, trades_df, initial_capital=10000.0):
    # hétvégén nincs kereskedés, ezért a hiányzó napokat eldobjuk (nem ffill),
    # mert a 0 hozamú hétvégék mesterségesen csökkentenék a volatilitást
    daily_equity = equity_series.resample("1D").last().dropna()
    daily_ret = daily_equity.pct_change().dropna()

    total_days = max((equity_series.index[-1] - equity_series.index[0]).days, 1)
    total_years = max(total_days / 365.25, 0.1)

    final_equity = equity_series.iloc[-1]
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0
    if final_equity > 0:
        cagr = ((final_equity / initial_capital) ** (1.0 / total_years) - 1.0) * 100.0
    else:
        cagr = -100.0

    peaks = equity_series.cummax()
    drawdowns = (equity_series - peaks) / peaks
    max_drawdown_pct = drawdowns.min() * 100.0

    mean_d = daily_ret.mean()
    std_d = daily_ret.std()
    ann_vol = std_d * np.sqrt(252) * 100.0 if std_d > 0 else 0.0

    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0
    downside_ret = daily_ret[daily_ret < 0]
    downside_std = np.sqrt((downside_ret ** 2).mean()) if len(downside_ret) > 0 else 0.0
    sortino = (mean_d / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    calmar = (cagr / abs(max_drawdown_pct)) if max_drawdown_pct < 0 else 0.0

    if len(trades_df) > 0:
        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        win_rate = len(wins) / total_trades * 100.0
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-8
        profit_factor = gross_profit / gross_loss
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    else:
        total_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        payoff_ratio = 0.0

    return {
        "Total Return (%)": round(total_return_pct, 2),
        "CAGR (%)": round(cagr, 2),
        "Annualized Volatility (%)": round(ann_vol, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Max Drawdown (%)": round(max_drawdown_pct, 2),
        "Calmar Ratio": round(calmar, 2),
        "Total Trades": total_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2),
        "Payoff Ratio": round(payoff_ratio, 2),
    }


def get_dd(series):
    peaks = series.cummax()
    return (series - peaks) / peaks * 100.0


def main():
    print("[*] Eredmények betöltése (US500)...")
    if not WF_RESULTS_FILE.exists():
        raise FileNotFoundError(f"Nem található: {WF_RESULTS_FILE}")

    results_df = pd.read_csv(WF_RESULTS_FILE)
    results_df["time"] = pd.to_datetime(results_df["time"])

    # a hiányzó ohlc/spread oszlopokat az alapadatokból pótoljuk
    print(f"[*] US500 alapadatok beemelése innen: {DATA_FILE}")
    base_df = pd.read_csv(DATA_FILE)
    base_df["time"] = pd.to_datetime(base_df["time"])

    need_cols = [c for c in ["open", "high", "low", "spread"] if c not in results_df.columns and c in base_df.columns]
    results_df = pd.merge(results_df, base_df[["time"] + need_cols], on="time", how="left")

    if not {"high", "low"}.issubset(results_df.columns):
        raise ValueError("A high/low oszlopok hiányoznak a US500 adatokból.")

    initial_capital = 1000.0
    sim_kwargs = dict(
        initial_capital=initial_capital,
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

    # xgboost
    print("[*] XGBoost Long-Only szimuláció (US500)...")
    xgb_equity, xgb_trades = simulate_trades(results_df, model_name="xgb", **sim_kwargs)
    xgb_metrics = calculate_metrics(xgb_equity, xgb_trades, initial_capital)
    xgb_trades.to_csv(REPORTS_DIR / "trades_xgboost_us500.csv", index=False)

    # lstm
    print("[*] LSTM Long-Only szimuláció (US500)...")
    lstm_equity, lstm_trades = simulate_trades(results_df, model_name="lstm", **sim_kwargs)
    lstm_metrics = calculate_metrics(lstm_equity, lstm_trades, initial_capital)
    lstm_trades.to_csv(REPORTS_DIR / "trades_lstm_us500.csv", index=False)

    # buy and hold összehasonlításnak
    bnh_equity = initial_capital * (results_df["close"] / results_df["close"].iloc[0])
    bnh_series = pd.Series(bnh_equity.values, index=results_df["time"])
    bnh_metrics = calculate_metrics(bnh_series, pd.DataFrame(), initial_capital)

    # összesítő táblázat
    summary_df = pd.DataFrame({
        "Mutató (Metrika)": list(xgb_metrics.keys()),
        "XGBoost US500": list(xgb_metrics.values()),
        "LSTM US500": list(lstm_metrics.values()),
        "Buy & Hold US500": list(bnh_metrics.values()),
    })

    metrics_csv = REPORTS_DIR / "backtest_metrics_comparison_us500.csv"
    summary_df.to_csv(metrics_csv, index=False)

    print("\n" + "=" * 78)
    print("VISSZATESZTELÉSI EREDMÉNYEK - US500 (S&P 500) H1 ML STRATÉGIA (LONG-ONLY)")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print("=" * 78)

    # equity görbe
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(xgb_equity.index, xgb_equity.values, label=f"XGBoost (Sharpe: {xgb_metrics['Sharpe Ratio']}, Hozam: {xgb_metrics['Total Return (%)']}%)", color="#1f77b4", linewidth=1.8)
    plt.plot(lstm_equity.index, lstm_equity.values, label=f"LSTM (Sharpe: {lstm_metrics['Sharpe Ratio']}, Hozam: {lstm_metrics['Total Return (%)']}%)", color="#ff7f0e", linewidth=1.8)
    plt.plot(bnh_series.index, bnh_series.values, label=f"Buy & Hold (Sharpe: {bnh_metrics['Sharpe Ratio']}, Hozam: {bnh_metrics['Total Return (%)']}%)", color="#7f7f7f", linestyle="--", alpha=0.7)
    plt.title("Tőkenövekedés (Equity Curve) - US500 H1 ML Long-Only", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Dátum", fontsize=11)
    plt.ylabel("Tőke (USD)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "equity_curves_us500.png")
    plt.close()

    # drawdown görbe
    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(xgb_equity.index, get_dd(xgb_equity), label="XGBoost DD", color="#1f77b4", linewidth=1.2)
    plt.plot(lstm_equity.index, get_dd(lstm_equity), label="LSTM DD", color="#ff7f0e", linewidth=1.2)
    plt.plot(bnh_series.index, get_dd(bnh_series), label="Buy & Hold DD", color="#7f7f7f", linestyle="--", alpha=0.6)
    plt.fill_between(xgb_equity.index, get_dd(xgb_equity), 0, color="#1f77b4", alpha=0.1)
    plt.title("Portfólió Tőke-visszaesés (Underwater Drawdown %) - US500", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Dátum", fontsize=11)
    plt.ylabel("Drawdown %", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "drawdown_curves_us500.png")
    plt.close()
    print("[ok] US500 ábrák és metrikák elmentve a reports/ mappába!\n")


if __name__ == "__main__":
    main()
