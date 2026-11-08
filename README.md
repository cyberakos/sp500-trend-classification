# Pénzügyi idősorok trendjeinek osztályozása gépi tanulási és mélytanulási modellekkel

A szakdolgozat empirikus részének forráskódja és eredményfájljai: XGBoost és LSTM összehasonlítása az S&P 500 index (US500 CFD) órás adatain, Triple Barrier címkézéssel, gördülőablakos Walk-Forward validációval és költségekkel terhelt backtesttel.

> Grama Ákos, Budapesti Gazdaságtudományi Egyetem, Pénzügy és Számvitel Kar, 2026. A módszertan és az eredmények részletes leírása és értelmezése a szakdolgozatban található. A kód kutatási célú, nem befektetési tanácsadás.

---

## Tartalom

| Mappa / fájl | Tartalom |
|---|---|
| `data_export.py` | Nyers H1 adatok letöltése MetaTrader 5-ből |
| `src/` | Jellemzők, címkézés, hangolás, modellek, Walk-Forward, backtest, összehasonlítások |
| `src/models/` | `XGBoostModel` és `LSTMModel` (PyTorch) osztályok |
| `reports/` | Eredménytáblák (CSV), kötésnaplók, ábrák |
| `reports/fixed_horizon/` | A fix horizontos kontrollkísérlet (RQ2) kimenetei |
| `Control_XAUUSD/` | Ugyanez a folyamat a spot arany adatsorán, saját README-vel |

A nyers és a feldolgozott adatfájlok méretük és a forrás felhasználási feltételei miatt nem részei a repozitóriumnak; a `data_export.py` futtatásával előállíthatók.

### Az `src/` szkriptjei

| Szkript | Feladat |
|---|---|
| `config.py` | Elérési utak és paraméterek |
| `features.py` | 20 stacionárius jellemző számítása |
| `labeling.py` | Triple Barrier címkék (±2,5 × ATR, 24 órás időkorlát) |
| `tune_optuna.py` | Hiperparaméter-keresés az első tanítóablakon |
| `train_single_split.py` | Egyszeri 70/15/15 felosztás, modellmentés |
| `feature_importance.py` | XGBoost gain és LSTM permutációs fontosság |
| `walk_forward.py` | 29 ablakos gördülő Walk-Forward validáció |
| `backtest.py` | Long-only kereskedési szimuláció és mutatók |
| `random_baseline.py` | Véletlen tippelés referenciája |
| `random_entry_benchmark.py` | Véletlen belépésű összehasonlítás |
| `labeling_fixed.py`, `walk_forward_fixed.py`, `compare_labeling.py` | Fix horizontos kontrollkísérlet (RQ2) |

---

## Telepítés

Python 3.12 ajánlott. A `MetaTrader5` csomag csak Windowson érhető el, és csak az adatletöltéshez kell.

```powershell
python -m venv venv
.\venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU esetén; CPU-n kihagyható
pip install -r requirements.txt
```

## Futtatás

A projekt gyökeréből, ebben a sorrendben:

```powershell
python .\data_export.py               # a kimenetet helyezd a data\raw\ mappába
python .\src\features.py
python .\src\labeling.py
python .\src\tune_optuna.py
python .\src\train_single_split.py
python .\src\feature_importance.py
python .\src\walk_forward.py
python .\src\backtest.py
python .\src\random_baseline.py
python .\src\random_entry_benchmark.py   # kb. 10-20 perc

# RQ2: fix horizontos kontrollkísérlet (a Triple Barrier eredményeket nem írja felül)
python .\src\labeling_fixed.py
python .\src\walk_forward_fixed.py
python .\src\compare_labeling.py
```

---

## Fő beállítások

| Terület | Érték |
|---|---|
| Adat | US500 H1, 2016-09-01 – 2026-08-31, MT5 szerveridő (UTC+2/+3) |
| Címkézés | ±2,5 × ATR(14), 24 gyertyás időkorlát |
| Walk-Forward | 15 000 gyertyás tanítóablak, 1 500 gyertyás teszt, 24 gyertyás kiürítés, 29 ablak |
| Hangolás | Optuna TPE (seed 42), az első 14 976 gyertyán; 30 (XGBoost) és 12 (LSTM) próbálkozás |
| XGBoost | 150 fa, mélység 5, η = 0,066, subsample 0,8, colsample 0,7, λ = 1,26, gamma = 1,0 (rögzített) |
| LSTM | 64 rejtett egység, 2 réteg, dropout 0,40, η = 0,00042, batch 256, 50 epoch |
| Osztályozási küszöb | τ = 0,42, δ = 0,10 |
| Backtest | belépés: p_buy ≥ 0,52 és p_buy − p_sell > 0,10; 1% kockázat; SL 2,0 × ATR; TP 2,5 × ATR; 24 órás időkorlát; 12 órás várakozás; platform-spread + 0,15 pont csúszás |

## Fő eredmények (out-of-sample, 2019-03-22 – 2026-08-28)

| | XGBoost | LSTM | Buy & Hold |
|---|---|---|---|
| Macro-F1 (argmax) | 0,348 | 0,354 | – |
| Teljes hozam (%) | 123,42 | 35,11 | 170,26 |
| Sharpe-ráta | 0,83 | 0,39 | 0,79 |
| Max. visszaesés (%) | −23,74 | −17,48 | −35,67 |
| Véletlen belépésnél jobb futások aránya | 99,0% | 78,0% | – |

A véletlen tippelés macro-F1 értéke 0,318. A részletes eredmények a `reports/` mappa CSV-fájljaiban, az értelmezésük a szakdolgozatban található.

## Ismert korlátok

- A napszaki jelzők (`is_asia`, `is_london`, `is_ny`, `is_overlap`) határai a `config.py`-ban UTC-ként vannak megadva, az adat viszont szerveridős, ezért a sávok 2–3 órával korábbiak a valós kereskedési szakaszoknál.
- Az LSTM bemenetei nincsenek skálázva.
- Minden futás egyetlen véletlenmaggal készült; GPU-n az LSTM eredményei futásonként kis mértékben eltérhetnek.
- A backtestben a belépés a jelzést adó gyertya záróárán történik, a tőkegörbe a lezárt ügyleteket követi, tőkeáttétel-korlát nincs.

---

## Szerző és licenc

Grama Ákos, BGE Pénzügy és Számvitel Kar, Gazdaságinformatikus BSc, Pénzintézeti informatikus specializáció. Témavezető: Dr. Buza Krisztián Antal. MIT License.
