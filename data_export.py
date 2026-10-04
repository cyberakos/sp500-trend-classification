import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import os

mt5.initialize()


symbol = "US500"
timezone = pytz.timezone("Etc/UTC")
start_date = datetime(2016, 9, 1, tzinfo=timezone)
end_date = datetime(2026, 8, 31, 23, 59, tzinfo=timezone)

mt5.symbol_select(symbol)

# Letöltendő idősíkok szótára
timeframes = {
    "H1":  mt5.TIMEFRAME_H1
}

for tf_name, tf_value in timeframes.items():
    print(f"{tf_name} adatok lekérése...")
    rates = mt5.copy_rates_range(symbol, tf_value, start_date, end_date)
    
    
    # DataFrame összeállítása
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Csak a releváns oszlopok megtartása és átnevezése
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]
    
    # Fájlmentés
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", f"data_{symbol.lower()}_{tf_name.lower()}.csv")
    df.to_csv(filepath, index=False)
    
    print(f"-> {filepath} elmentve | Sorok száma: {len(df):,} | Időszak: {df['time'].iloc[0]} - {df['time'].iloc[-1]}")

# 4. Terminál kapcsolat bontása
mt5.shutdown()
print("Az adatok letöltése és mentése sikeresen befejeződött.")