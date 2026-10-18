# Kontrollkísérlet: spot arany (XAUUSD) H1

A fő projekt változatlan kódja, a spot arany órás adatsorán futtatva (2016-09-01 – 2026-08-31).
A szakdolgozat 4.2.3. alfejezetének és a 4.3.3. alfejezet arany-eredményeinek forrása.

## Eltérések a fő projekttől

- `data_export.py`: XAUUSD szimbólum. A kimeneti fájlnév szándékosan `data_US500_h1.csv` marad,
  hogy a többi szkript változtatás nélkül fusson. A `*_us500*` nevű fájlok ebben a mappában
  arany adatot tartalmaznak.
- `clean_gold_data.py`: új lépés az export után. Törli a 2021 tavaszáig előforduló 00:00-s napi
  összesítő gyertyákat (1 142 sor), mert ezek záróára a nap végi ár volt, vagyis jövőbeli
  információt tartalmaztak. A 2021. január–szeptember közötti nulla spreadeket (4 171 sor) az adott
  óra nem nulla spreadjeinek mediánjával pótolja. Az eredeti adatot a
  `data/raw/data_xauusd_h1_nyers.csv` fájlba menti.
- A hiperparaméterek az arany saját első tanítóablakán lettek újrahangolva.
- Az RQ2 szkriptek (`labeling_fixed.py`, `walk_forward_fixed.py`, `compare_labeling.py`) a mappában
  megvannak, de itt nem futnak, mert az US500 fájlokra épülnek.

## Futtatás

A MetaTrader 5 terminálnak futnia kell. A lépések a mappa gyökeréből, egymás után futtathatók:

```powershell
python.exe .\data_export.py               # XAUUSD letöltése; a kimenetet helyezd át: data\raw\
python.exe .\clean_gold_data.py           # napi összesítő gyertyák törlése, nulla spreadek pótlása
python.exe .\src\features.py
python.exe .\src\labeling.py
python.exe .\src\tune_optuna.py
python.exe .\src\train_single_split.py
python.exe .\src\feature_importance.py
python.exe .\src\walk_forward.py
python.exe .\src\backtest.py
python.exe .\src\random_baseline.py       # véletlen tippelés referenciája
python.exe .\src\random_entry_benchmark.py  # véletlen belépésű összehasonlítás
```

A `clean_gold_data.py`-t közvetlenül az export után, egyszer futtasd. Ha a már tisztított fájlon
újra lefut, a `data_xauusd_h1_nyers.csv` mentést is a tisztított adat írja felül.

A `data/`, a `models/saved/` és az Optuna-adatbázis a `.gitignore` miatt nincs a repóban,
ezért a futtatás a letöltéssel kezdődik.

## Eredmények

Out-of-sample időszak: 2019-04-04 – 2026-08-28, 43 727 óra, 29 ablak.
Osztályeloszlás: Long 42,75%, Short 37,91%, Semleges 19,34%.

| Modell / viszonyítási alap | Pontosság | Macro-F1 |
|---|---|---|
| Véletlen tippelés (50 futás átlaga) | 33,28% | 0,324 |
| Mindig Long | 42,75% | 0,200 |
| XGBoost, küszöbös (τ = 0,42, δ = 0,10) | 32,46% | 0,320 |
| LSTM, küszöbös (τ = 0,42, δ = 0,10) | 34,44% | 0,334 |
| XGBoost, argmax | 41,40% | 0,371 |
| LSTM, argmax | 40,28% | 0,356 |

| Mutató | XGBoost | LSTM | Buy & Hold |
|---|---|---|---|
| Total Return (%) | 138,51 | 98,98 | 246,97 |
| Sharpe-ráta | 0,95 | 0,77 | 1,03 |
| Max Drawdown (%) | -18,49 | -27,73 | -28,76 |
| Kötésszám | 1 046 | 1 100 | – |
| Profit Factor | 1,20 | 1,14 | – |

Véletlen belépésű összehasonlítás (azonos szabályok és kötésszám, 200 futás): az XGBoost hozama
a véletlen futások 99,5%-ánál, az LSTM-é 95,5%-ánál magasabb.

A hibás 00:00-s gyertyák törlése előtt az LSTM macro-F1 értéke az első nyolc ablakban 0,548 volt,
a backtest hozama pedig 762,92%. Ezek az eredmények az adathibából adódtak, és nem érvényesek.
