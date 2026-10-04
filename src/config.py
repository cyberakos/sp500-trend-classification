from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # eszköz és fájl beállítások
    symbol: str = "US500"
    timeframe: str = "H1"
    raw_data_path: str = "data/raw/data_us500_h1.csv"
    processed_dir: str = "data/processed"

    # feature engineering paraméterek
    ema_fast_period: int = 50
    ema_slow_period: int = 200
    # a tanh telítődését szabályozó osztó
    tanh_dist_multiplier: float = 2.5
    atr_period: int = 14
    rsi_period: int = 14
    vol_sma_period: int = 20
    return_lags: tuple = (1, 3, 6, 24)

    # szesszió határok utc-ben (az index hétfőtől péntekig megy, hétvégén nincs)
    asia_start: float = 0.0
    asia_end: float = 8.0
    london_start: float = 8.0
    london_end: float = 16.5
    ny_start: float = 13.5
    ny_end: float = 21.0
    overlap_start: float = 13.5
    overlap_end: float = 16.5

    # triple barrier címkézés
    barrier_horizon: int = 24        # kb. egy kereskedési nap (24 óra)
    barrier_multiplier: float = 2.5  # atr szorzó a felső és alsó sávhoz

    # walk-forward beállítások
    # h1-en kb. 23 bár/nap * 252 nap = ~5800 bár egy évben
    train_bars: int = 15000          # kb. 2.6 év tanításra
    val_bars: int = 2400             # kb. 5 hónap hangolásra
    test_bars: int = 1500            # kb 3 hónap out of sample teszt
    step_bars: int = 1500            # ugyanannyit lépünk mint a teszt hossza
    purge_bars: int = 24             # a címkék előre nézése miatti biztonsági rés
