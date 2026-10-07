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
- **Szivárgásmentes hangolás:** az Optuna csak az első Walk-Forward tanítóablakon belül keres hiperparamétereket, így a paraméterválasztás nem lát rá a tesztidőszakra.
- **Kontrollkísérlet (RQ2):** ugyanaz a csővezeték fix horizontos címkékkel is lefuttatható, így a Triple Barrier címkézés hatása elkülöníthető.

---

## 2. Projektstruktúra

```text
.
│   data_export.py                          # Nyers ár- és forgalmi adatok exportálása MetaTrader 5-ből
│   requirements.txt                        # A projekt Python-függőségei
│   futtatas_rq2.bat                        # Az RQ2 kontrollkísérlet futtatása egy lépésben (Windows)
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
│           labeled_fixed_us500_h1.csv      # Fix horizontos kontrollcímkék (RQ2)
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
│   │   best_hyperparameters_us500_backup.json  # Az előző hangolás paraméterei (automatikus mentés)
│   │   optuna_us500.db                         # Az Optuna próbálkozásai (SQLite)
│   │   feature_importance_comparison_us500.csv # XGBoost és LSTM jellemzőfontosság
│   │   trades_lstm_us500.csv                   # LSTM szimulált kereskedési naplója
│   │   trades_xgboost_us500.csv                # XGBoost szimulált kereskedési naplója
│   │   walk_forward_results_us500.csv          # 29 Walk-Forward tesztablak out-of-sample predikciói
│   │
│   ├───fixed_horizon                       # Az RQ2 kontrollkísérlet kimenetei (WF, backtest, összehasonlító táblák)
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
    │   labeling_fixed.py                   # Fix horizontos kontrollcímkézés (RQ2)
    │   walk_forward_fixed.py               # WF és backtest a fix horizontos címkékkel (RQ2)
    │   compare_labeling.py                 # Triple Barrier és fix horizont összehasonlítása (RQ2)
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

# GPU-s PyTorch (CUDA 12.1); CPU-n ez a sor kihagyható
pip install torch --index-url https://download.pytorch.org/whl/cu121

# A többi csomag
pip install -r requirements.txt
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

Az RQ2 kontrollkísérlet a fenti lépések után futtatható. A meglévő Triple Barrier eredményeket nem írja felül, a kimenetek a `reports/fixed_horizon/` mappába kerülnek:

```powershell
python.exe .\src\labeling_fixed.py       # fix horizontos címkék, ugyanazokon a sorokon
python.exe .\src\walk_forward_fixed.py   # Walk-Forward és backtest a fix horizontos címkékkel
python.exe .\src\compare_labeling.py     # összehasonlító táblák és közös tőkegörbe
```

Ugyanez egy lépésben: `futtatas_rq2.bat` (szükség esetén létrehozza a virtuális környezetet és telepíti a függőségeket).

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
| Hiperparaméter-hangolás | az első 14 976 gyertya (2016-09-13 – 2019-03-21), 80/20 tanító/validációs, 24 gyertyás hézag; TPE, seed = 42; 30 (XGBoost) és 12 (LSTM) próbálkozás, cél: validációs macro-F1 |
| Hiperparaméterek | XGBoost: 150 fa, mélység 5, η = 0,066, subsample 0,8, colsample 0,7, λ = 1,26; LSTM: 64 rejtett egység, 2 réteg, dropout 0,40, η = 0,00042, batch 256 |
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
| Véletlen tippelés (50 futás átlaga) | 33,3% | 0,318 |
| Mindig Long | 43,58% | 0,202 |
| XGBoost, küszöbös predikció (τ = 0,42, δ = 0,10) | 31,60% | 0,309 |
| LSTM, küszöbös predikció (τ = 0,42, δ = 0,10) | 31,43% | 0,311 |
| XGBoost, argmax | 42,99% | 0,348 |
| LSTM, argmax | 42,21% | 0,354 |

- Küszöbös predikcióval mindkét modell macro-F1-e a véletlen tippelés szintjén vagy kissé alatta van. Argmax-predikcióval kissé a véletlen fölött vannak, a pontosságuk viszont a mindig Long viszonyítási alap alatt marad.
- A két modell között nincs érdemi különbség: az ablakonkénti macro-F1 átlaga 0,306 (XGBoost) és 0,304 (LSTM), az LSTM a 29 ablakból 11-ben jobb, páros Wilcoxon-próba p ≈ 0,47. Az ablakok tanítóadata átfedi egymást, ezért a p-érték tájékoztató jellegű.
- A backtest belépési szabályát (p_buy ≥ 0,52, p_buy − p_sell > 0,10) teljesítő órák után az ár az esetek 44,8%-ában (XGBoost) és 46,0%-ában (LSTM) érte el először a felső korlátot, szemben a 43,6%-os alaprátával.

### 6.2. Szimulált stratégia (Long-only, költségekkel)

| Mutató | XGBoost | LSTM | Buy & Hold |
|---|---|---|---|
| Total Return (%) | 123,42 | 35,11 | 170,26 |
| CAGR (%) | 11,42 | 4,13 | 14,31 |
| Évesített szórás (%) | 14,06 | 12,25 | 19,18 |
| Sharpe-ráta | 0,83 | 0,39 | 0,79 |
| Sortino-ráta | 0,73 | 0,30 | 0,75 |
| Max Drawdown (%) | -23,74 | -17,48 | -35,67 |
| Calmar-ráta | 0,48 | 0,24 | 0,40 |
| Kötésszám | 1 259 | 974 | – |
| Találati arány (%) | 48,53 | 47,13 | – |
| Profit Factor | 1,12 | 1,06 | – |
| Payoff Ratio | 1,19 | 1,18 | – |

- Az XGBoost-stratégia hozama a Buy & Hold alap alatt van, a szórása és a maximális visszaesése alacsonyabb. A Sharpe-ráta a Buy & Hold fölött van, de a két érték nem vethető össze közvetlenül (lásd 7. pont: a stratégia tőkegörbéje csak a lezárt ügyleteknél változik). A stratégia az out-of-sample órák kb. 30%-ában tart nyitott pozíciót.
- Az LSTM-stratégia pozitív, de alacsony hozamot ért el, kb. 24%-os piaci kitettséggel.
- Kilépések: XGBoost 603 stop-loss, 539 profitcél, 117 időkorlát (átlagos tartás 10,5 gyertya); LSTM 486 stop-loss, 410 profitcél, 78 időkorlát (átlagos tartás 10,8 gyertya).

Évenkénti eredmény az induló tőke százalékában (a lezárt ügyletek PnL-jének összege a kilépés éve szerint):

| Év | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 7,7 | 40,4 | 6,9 | -24,6 | 58,1 | 24,3 | -2,5 | 13,1 |
| LSTM | 10,0 | 2,5 | 8,2 | -8,7 | 17,1 | 0,5 | 14,1 | -8,6 |

Az XGBoost-stratégia nettó eredményének nagy része 2020-ból és 2023-ból származik, 2022-ben a stratégia jelentős veszteséget szenvedett.

### 6.3. Jellemzőfontosság

A jellemzőfontosság az egyszeri felosztáson (70/15/15) tanított modelleken készül, a teszt-15%-on (2025–2026).

- XGBoost (gain): a legnagyobb súlyú változók az `is_london` (22,1%), a `hour_cos` (12,4%) és az `is_asia` (5,8%). Az óra- és naptípusú változók, valamint a napszaki jelzők együtt kb. 61%-ot adnak, a hozam-, trend- és volatilitásjellemzők egyenként legfeljebb 4,6%-ot.
- LSTM (permutációs fontosság, macro-F1 csökkenése): a `rsi` (19,3%), a `hour_cos` (17,3%), a `day_sin` (12,2%), az `is_london` (9,1%) és a `hour_sin` (9,0%) a legfontosabb; az idő- és szakaszváltozók együtt kb. 66%-ot adnak. A log-hozamok, a `body_pct` és az `atr_pct` permutálása nem rontott a teljesítményen.
- Az `is_weekend` értéke az adatsorban végig 0, a fontossága mindkét modellnél 0.

A modellek tehát jelentős részben napszaki mintázatra támaszkodnak. Ennek a hatásnak a mértéke a jelenlegi elemzésből nem választható el a trendinformációtól.

Az egyszeri felosztású LSTM a validációs veszteség alapján a 4. korszaknál állt meg, és a tesztrészen szinte kizárólag vételt jósol (argmax szerint a tesztórák kb. 96%-ában), a küszöbös szabályt pedig egyetlen órán sem teljesíti. A permutációs fontosság ezért egy majdnem állandó kimenetű modell apró eltéréseit méri, és csak óvatosan értelmezhető.

---

## 7. Ismert korlátok

- **Hiperparaméter-hangolás:** a paraméterek a 2016–2019-es időszakon lettek kiválasztva, és változatlanul érvényesek mind a 29 ablakban, ezért a későbbi piaci szakaszokban nem feltétlenül optimálisak. A legjobb érték két paraméternél a keresési tér szélére esett (famélység 5, dropout 0,40). A hangolás egy korábbi változata az adatsor nagyobb részén futott, és átfedett a tesztidőszakkal; a jelenlegi eredmények már a javított, szivárgásmentes hangolással készültek.
- **Eltérő tanítási feltételek:** a Walk-Forward ciklusokban az LSTM belső validációs szelet és korai leállás nélkül, rögzített epochszámon át tanul, az XGBoost viszont regularizált. Az összehasonlítás torzításának iránya nem ismert.
- **Eltérő küszöbök:** az osztályozási kiértékelés (τ = 0,42, δ = 0,10) és a backtest (τ = 0,52, δ = 0,10) küszöbei nem azonosak.
- **Pozícióméret és tőkeáttétel:** a pozícióméret az 1% kockázatból és a stop-loss távolságából adódik, tőkeáttétel-korlát nélkül. A kötésnapló alapján a pozíciók névértéke a tőke mediánban kb. 2,3-szerese (LSTM: 2,0), legfeljebb kb. 8-szorosa, a kötések kb. 90%-ában meghaladja a tőkét.
- **Backtest-egyszerűsítések:** az SL/TP kilépéseknél külön spread nem kerül levonásra, a belépés a jelet adó gyertya záróárán történik, a tőkegörbe a lezárt ügyletek realizált eredményét mutatja, szemben a minden gyertyán értékelt Buy & Hold alappal.
- **Napszaki jelzők:** a szerveridő fix ablakai az amerikai tőzsdei főszakaszt csak részben fedik le.
- **Egy eszköz, egy randomseed:** az eredmények szórása nem ismert, és az általánosíthatóság nem igazolt.

---

## 8. Generált kimenetek

- `reports/best_hyperparameters_us500.json`: az Optuna legjobb XGBoost- és LSTM-paraméterei.
- `reports/optuna_us500.db`: az Optuna próbálkozásai; ebből az ábrák hangolás nélkül is újrarajzolhatók.
- `reports/fixed_horizon/`: az RQ2 kontrollkísérlet Walk-Forward- és backtest-eredményei, valamint az összehasonlító táblák (`rq2_*.csv`) és a közös tőkegörbe.
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

