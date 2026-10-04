# Pénzügyi idősorok trendjeinek osztályozása gépi tanulási és mélytanulási modellekkel

Kutatási és gépi tanulási keretrendszer az **S&P 500 index (US500 CFD)** órás (H1) idősíkú trendjeinek osztályozására és szimulált piaci végrehajtására. A projekt az **XGBoost** és az **LSTM** modellt hasonlítja össze Triple Barrier címkézéssel, gördülőablakos Walk-Forward validációval és költségekkel terhelt visszatesztelés (backtest) segítségével.

> A repó a BGE Pénzügy és Számvitel Kar szakdolgozatának empirikus része (Grama Ákos, 2026). A számítások kutatási célúak, nem befektetési tanácsadást jelentenek.

---

## 1. A projekt áttekintése

A pénzügyi idősorokon tanuló modellek gyakori hibái a túlilleszkedés (overfitting), a jövőbe tekintési torzítás (lookahead bias) és a tranzakciós költségek figyelmen kívül hagyása. A csővezeték ezekre az alábbi megoldásokat alkalmazza:

- **Stacionárius jellemzőkészlet:** 20 magyarázó változó, amely több horizontú logaritmikus hozamokat, gyertyamorfológiai mutatókat, volatilitással normalizált trendtávolságokat (`tanh`), RSI-t, normalizált ATR-t, relatív forgalmat, trigonometrikus időkódolást és napszaki jelzőket tartalmaz. A nyers árak nem kerülnek a modellek bemenetére.
- **Dinamikus Triple Barrier címkézés:** a profitcél és a stop-loss 2,5 × ATR(14) távolságra van a belépéstől, a függőleges (időbeli) korlát 24 gyertya. Az egyidejű átlépés a neutrális (0) osztályba kerül.
- **Idősoros validáció:** 29 ablakos, gördülő Walk-Forward felosztás 24 gyertyás kiürítési hézaggal (purging) a tanító- és tesztadat között, így az átfedő címkék nem okoznak szivárgást.
- **Költségekkel terhelt backtest:** eseményvezérelt, Long-only szimuláció, ügyletenként 1% kockázattal, a platform spread-adatával, fix csúszással (slippage) és gap-kezeléssel.

---

## 2. Projektstruktúra

```text
.
│   data_export.py                          # Nyers ár- és forgalmi adatok exportálása MetaTrader 5-ből
│   run_pipeline.py                         # A lépések futtatása egyben vagy részenként (--only)
│   map.txt                                 # Projekt könyvtártérkép
│   README.md                               # Rendszerdokumentáció
│
├───data
│   ├───raw
│   │       data_us500_h1.csv               # Nyers H1 gyertyaadatok (OHLC, tick volume, spread)
│   │
│   └───processed
│           features_us500_h1.csv           # Generált stacionárius magyarázó változók
│           labeled_us500_h1.csv            # Triple Barrier címkékkel ellátott adatkészlet
│           walk_forward_results.csv        # Korábbi validációs eredmények biztonsági mentése
│
├───models
│   └───saved
│           lstm_single_split_us500.pt      # Egyszeri felosztáson tanított LSTM súlyok (PyTorch)
│           xgboost_single_split_us500.json # Egyszeri felosztáson tanított XGBoost modell
│
├───reports                                 # Numerikus jelentések, táblázatok és kereskedési naplók
│   │   backtest_metrics_comparison_us500.csv   # Gazdasági és kockázati mutatók összehasonlítása
│   │   best_hyperparameters_us500.json         # Optuna által talált hiperparaméterek
│   │   feature_importance_comparison_us500.csv # XGBoost és LSTM jellemzőfontosság
│   │   trades_lstm_us500.csv                   # LSTM szimulált kereskedési naplója
│   │   trades_xgboost_us500.csv                # XGBoost szimulált kereskedési naplója
│   │   walk_forward_results_us500.csv          # 29 Walk-Forward tesztablak out-of-sample predikciói
│   │
│   └───figures                             # Ábrák
│           confusion_matrices_single_us500.png
│           drawdown_curves_us500.png
│           equity_curves_us500.png
│           lstm_feature_importance_us500.png
│           lstm_learning_curves_us500.png
│           optuna_lstm_importance_us500.png
│           optuna_xgb_importance_us500.png
│           xgb_feature_importance_us500.png
│           xgb_loss_curve_us500.png
│
└───src                                     # Forráskód
    │   __init__.py
    │   config.py                           # Elérési utak és paraméterek
    │   features.py                         # Indikátorszámítás és stacionárius transzformációk
    │   labeling.py                         # Triple Barrier címkézés
    │   tune_optuna.py                      # Bayes-i hiperparaméter-optimalizálás (Optuna)
    │   train_single_split.py               # 70/15/15 egyszeri felosztás, modellmentés
    │   feature_importance.py               # XGBoost gain és LSTM permutációs fontosság
    │   walk_forward.py                     # Gördülőablakos Walk-Forward validáció
    │   backtest.py                         # Piaci szimulátor és portfóliómetrikák
    │
    └───models
            __init__.py                     # Exportálja az XGBoostModel és LSTMModel osztályokat
            xgboost_model.py                # XGBoost wrapper
            lstm_model.py                   # PyTorch LSTM, adatbetöltő és tanítás
```

A `data_export.py` az adatokat a `data/` mappába menti. A `config.py` a `data/raw/data_us500_h1.csv` útvonalat várja, ezért az exportált fájlt a futtatás után át kell helyezni a `data/raw/` mappába.

---

## 3. Rendszerkövetelmények és telepítés

Ajánlott környezet: Python 3.12. A `MetaTrader5` csomag csak Windowson érhető el, és csak az adatletöltéshez kell. Ha a nyers CSV már megvan, a többi lépés más rendszeren is futtatható. Az LSTM CUDA-képes GPU-n gyorsabb, de CPU-n is fut.

```powershell
# Virtuális környezet létrehozása és aktiválása
python -m venv venv
.\venv\Scripts\activate          # Linux/macOS: source venv/bin/activate

# Szükséges csomagok
pip install numpy pandas matplotlib scikit-learn pytz
pip install xgboost optuna MetaTrader5
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. A csővezeték futtatása

A lépések egymásra épülnek, a projekt gyökeréből futtathatók:

```powershell
# 1. Nyers adatok letöltése a MetaTrader 5 terminálból (ha a data_us500_h1.csv még nincs meg)
python.exe .\data_export.py          # a kimenetet helyezd át: data\raw\

# 2. Jellemzők generálása
python.exe .\src\features.py

# 3. Triple Barrier címkék
python.exe .\src\labeling.py

# 4. Hiperparaméter-optimalizáció (létrehozza a best_hyperparameters_us500.json-t)
python.exe .\src\tune_optuna.py

# 5. Egyszeri felosztású modellek tanítása és mentése (models/saved)
python.exe .\src\train_single_split.py

# 6. Jellemzőfontossági elemzés
python.exe .\src\feature_importance.py

# 7. Walk-Forward validáció
python.exe .\src\walk_forward.py

# 8. Backtest és ábrák
python.exe .\src\backtest.py
```

---

## 5. Főbb beállítások

| Terület | Érték |
|---|---|
| Eszköz, idősík | US500 (S&P 500 CFD), H1, 2016-09-01 – 2026-08-31 |
| Nyers / címkézett rekordszám | 59 152 / 58 928 (200 gyertya bemelegedés, 24 gyertya levágva a horizont miatt) |
| Címkézés | ±2,5 × ATR(14), 24 gyertyás időkorlát |
| Walk-Forward | 15 000 gyertyás tanítóablak, 1 500 gyertyás teszt, 24 gyertyás kiürítés, 29 ablak |
| Out-of-sample időszak | 2019-03-22 – 2026-08-28 (43 928 gyertya) |
| Egyszeri felosztás | 70% tanító / 15% validációs / 15% teszt, 24 gyertyás hézagokkal |
| Döntési küszöbök (Walk-Forward) | τ = 0,42, δ = 0,10 |
| Backtest-belépés | p_buy ≥ 0,52, p_buy − p_sell > 0,10 |
| Backtest-kockázat | 1% / ügylet, SL = 2,0 × ATR, TP = 2,5 × ATR, 24 gyertyás időkorlát, 12 gyertyás várakozás |
| Költségek | a platform spread-adata (átlagosan kb. 0,45 pont; tartalék: 0,40 pont), 0,15 pont csúszás |
| Induló tőke | a százalékos mutatók az induló tőkétől függetlenek (a `backtest.py` paramétere) |

Az időbélyegek a MetaTrader 5 szerveridejében vannak (az amerikai óraátállást követik), a `hour` és `day_of_week` jellemzők és a napszaki jelzők erre vonatkoznak.

---

## 6. Eredmények (US500 H1)

### 6.1. Osztályozás (Walk-Forward, out-of-sample, 43 928 óra)

Az out-of-sample minta osztályeloszlása: Short 41,29%, Neutrális 15,14%, Long 43,58%.

| Modell / viszonyítási alap | Pontosság | Macro-F1 |
|---|---|---|
| Véletlen tippelés (200 futás átlaga) | 33,35% | 0,318 |
| Mindig Long | 43,58% | 0,202 |
| XGBoost, küszöbös predikció (τ = 0,42, δ = 0,10) | 32,77% | 0,318 |
| LSTM, küszöbös predikció (τ = 0,42, δ = 0,10) | 36,40% | 0,342 |
| XGBoost, argmax | 42,61% | 0,348 |
| LSTM, argmax | 41,68% | 0,355 |

- A küszöbös XGBoost macro-F1 értéke megegyezik a véletlenével, az LSTM-é 0,024-del magasabb. Az LSTM a 29 ablak közül 23-ban ért el jobb macro-F1-et (ablakonkénti átlag: 0,338 az LSTM-nél, 0,316 az XGBoost-nál; páros Wilcoxon-próba p ≈ 0,0003). Az ablakok tanítóadata átfedi egymást, ezért a p-érték tájékoztató jellegű.
- Argmax-predikcióval mindkét modell macro-F1-e kissé a véletlen fölött van, a pontosságuk viszont a mindig Long viszonyítási alap alatt marad.

### 6.2. Szimulált stratégia (Long-only, költségekkel)

| Mutató | XGBoost | LSTM | Buy & Hold |
|---|---|---|---|
| Total Return (%) | 105,13 | -1,81 | 170,26 |
| CAGR (%) | 10,14 | -0,25 | 14,31 |
| Évesített szórás (%) | 14,62 | 14,57 | 19,18 |
| Sharpe-ráta | 0,73 | 0,07 | 0,79 |
| Sortino-ráta | 0,65 | 0,06 | 0,75 |
| Max Drawdown (%) | -25,99 | -21,71 | -35,67 |
| Calmar-ráta | 0,39 | -0,01 | 0,40 |
| Kötésszám | 1 348 | 1 382 | – |
| Találati arány (%) | 48,44 | 46,24 | – |
| Profit Factor | 1,10 | 1,00 | – |
| Payoff Ratio | 1,17 | 1,16 | – |

- Az XGBoost-stratégia hozama és Sharpe-rátája a Buy & Hold alap alatt van, a szórása és a maximális visszaesése alacsonyabb. A stratégia az out-of-sample órák kb. 32%-ában tart nyitott pozíciót, ami önmagában csökkenti a kitettséget.
- Az LSTM-stratégia hozama lényegében nulla.
- Az XGBoost-stratégia kilépései: 661 stop-loss, 580 profitcél, 107 időkorlát; az átlagos tartási idő 10,4 gyertya.

Évenkénti eredmény az induló tőke százalékában (a lezárt ügyletek PnL-jének összege a kilépés éve szerint):

| Év | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 9,7 | 14,3 | 13,8 | -16,7 | 49,6 | 69,7 | -34,3 | -1,0 |
| LSTM | 6,1 | -7,1 | 1,1 | -12,7 | 13,2 | 4,5 | -13,2 | 6,4 |

Az XGBoost-stratégia teljes nettó eredménye a 2023–2024-es évekből származik, 2022-ben és 2025-ben a stratégia veszteséges volt.

### 6.3. Jellemzőfontosság

- XGBoost (gain): a legnagyobb súlyú változók az `is_london` (21,1%), az `hour_cos` (10,6%) és az `is_asia` (6,9%). Az óra- és naptípusú változók együtt kb. 59,7%-ot adnak, a hozam-, trend- és volatilitásjellemzők egyenként legfeljebb 4,5%-ot.
- LSTM (permutációs fontosság): a `rsi` (17,5%), a `day_sin` (16,0%), az `hour_sin` (15,2%) és az `is_london` (14,7%) a legfontosabb; az óra- és napváltozók, valamint a napszaki jelzők együtt kb. 61,9%-ot adnak.
- Az `is_weekend` értéke az adatsorban végig 0, a fontossága mindkét modellnél 0.

A modellek tehát jelentős részben napszaki mintázatra támaszkodnak. Ennek a hatásnak a mértéke a jelenlegi elemzésből nem választható el a trendinformációtól.

---

## 7. Ismert korlátok

- **Hiperparaméter-hangolás:** az Optuna az adatsor első 85%-án hangol, ezért a Walk-Forward tesztablakok jelentős része átfedett a hangolási mintával. Az eredmények a hiperparaméter-választás szempontjából nem tisztán out-of-sample jellegűek. A keresési tér szélére esett legjobb érték több paraméternél is (300 fa, 0,6 sormintavétel, 96 rejtett egység, 0,20 dropout).
- **Eltérő tanítási feltételek:** a Walk-Forward ciklusokban az LSTM belső validációs szelet és korai leállás nélkül, rögzített epochszámon át tanul, az XGBoost viszont regularizált. Az összehasonlítás torzításának iránya nem ismert.
- **Eltérő küszöbök:** az osztályozási kiértékelés (τ = 0,42, δ = 0,10) és a backtest (τ = 0,52, δ = 0,10) küszöbei nem azonosak.
- **Pozícióméret és tőkeáttétel:** a pozícióméret az 1% kockázatból és a stop-loss távolságából adódik, tőkeáttétel-korlát nélkül. A kötésnapló alapján a pozíciók névértéke a tőke mediánban kb. 2,2-szerese, legfeljebb kb. 8-szorosa, a kötések kb. 92%-ában meghaladja a tőkét.
- **Backtest-egyszerűsítések:** az SL/TP kilépéseknél külön spread nem kerül levonásra, a belépés a jelet adó gyertya záróárán történik, a tőkegörbe a lezárt ügyletek realizált eredményét mutatja, szemben a minden gyertyán értékelt Buy & Hold alappal.
- **Napszaki jelzők:** a szerveridő fix ablakai az amerikai tőzsdei főszakaszt csak részben fedik le.
- **Egy eszköz, egy randomseed:** az eredmények szórása nem ismert, és az általánosíthatóság nem igazolt.

---

## 8. Generált kimenetek

- `reports/best_hyperparameters_us500.json`: az Optuna legjobb XGBoost- és LSTM-paraméterei.
- `reports/walk_forward_results_us500.csv`: a 29 tesztablak out-of-sample predikciói és valószínűségei.
- `reports/trades_xgboost_us500.csv`, `reports/trades_lstm_us500.csv`: kötésenkénti napló (belépés, kilépés, árak, PnL, kilépési ok: TP / SL / TIME).
- `reports/backtest_metrics_comparison_us500.csv`: gazdasági és kockázati mutatók.
- `reports/feature_importance_comparison_us500.csv`: XGBoost (gain) és LSTM (permutációs) jellemzőfontosság.
- `reports/figures/`: tőkegörbék, visszaesési görbék, tanulási görbék, konfúziós mátrixok, jellemzőfontossági és Optuna-ábrák.

---

## 9. Szerző és licenc

- **Készítette:** Grama Ákos, Budapesti Gazdaságtudományi Egyetem, Pénzügy és Számvitel Kar, Gazdaságinformatikus BA/BSc, Pénzügyi informatikus specializáció
- **Témavezető:** Dr. Buza Krisztián Antal, egyetemi docens
- **Év:** 2026
- **Licenc:** MIT License

