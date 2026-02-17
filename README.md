# FCS-HSE-x-WB-Hackathon

**WB HACKATHON — SNEAKERS SALES PREDICTION
APPROACH SUMMARY**

ЗАДАЧА
------
Предсказать количество проданных единиц конкретного товара (кроссовки) за
один день при определённой цене. Тестовый период: 14 дней (2025-07-08 —
2025-07-21). Метрика: weighted MAE (wMAE), w=7 для дней с продажами, w=1 для
дней без продаж.

Лучший результат: wMAE = 0.370 (реплицируется через replicate_370.py)


КЛЮЧЕВЫЕ ИНСАЙТЫ ИЗ EDA
------------------------
1. qty ВСЕГДА кратен 3 (0, 3, 6, 9, ...) — товары продаются пачками/парами.
2. 87% записей имеют qty=0 — крайне разреженные данные (zero-inflated).
3. prev_leftovers — самый сильный предиктор (corr=0.374 с qty).
4. Промо удваивает продажи (mean_qty: 0.58 без промо vs 1.12 с промо).
5. Цена — слабый глобальный предиктор, но есть per-item price elasticity.
6. 947 из 949 тестовых товаров имеют историю в train (только 2 cold start).
7. Train: 309K записей, 2743 товара, 369 дней.
8. Test:  12.8K записей, 949 товаров, 14 дней.
9. Distribution shift: в тесте выше цены (+36%), больше промо (74% vs 49%),
   больше остатков (+71%).


АРХИТЕКТУРА РЕШЕНИЯ: HURDLE MODEL С ГРУППИРОВАННЫМИ ПОРОГАМИ
-------------------------------------------------------------

Решение реализовано в train_tabular_370.py и состоит из следующих этапов:

  ┌─────────────────────────────────────────────────────────────────┐
  │  ЭТАП 1: FEATURE ENGINEERING                                    │
  │                                                                 │
  │  Используется 69 фичей:                                         │
  │   - Исходные: price, is_promo, prev_leftovers,                  │
  │     sneakers_google_trends                                      │
  │   - Календарные: day_of_week, day_of_month, week_of_year,       │
  │     month, is_weekend, is_holiday, days_to_holiday              │
  │   - Item static: агрегаты по истории товара (mean, std, max,    │
  │     pct_nonzero, price stats, etc.)                             │
  │   - Price relative: цена относительно истории товара            │
  │   - Leftovers derived: log_leftovers, leftovers_vs_item_mean    │
  │   - Forward/backward delta: изменения остатков                  │
  │   - Sell rate: скорость продаж                                  │
  │   - Lag features: rolling stats по price/leftovers за 7/14 дней │
  │   - Promo features: promo_days_7d                               │
  │   - Market features: агрегаты по всему рынку                    │
  │   - Interaction features: комбинации ключевых предикторов       │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  ЭТАП 2: TIME-BASED 3-WAY SPLIT                                 │
  │                                                                 │
  │  Данные разбиваются на 3 части по времени:                      │
  │   - train_split: основная обучающая выборка                     │
  │   - tuning_split: для early stopping и выбора модели            │
  │   - val_split (holdout): для оптимизации порогов и финальной    │
  │     оценки wMAE                                                 │
  │                                                                 │
  │  Размеры сплитов настраиваются в config.py:                     │
  │   - tuning_days: количество дней для tuning split               │
  │   - val_days: количество дней для holdout split                 │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  ЭТАП 3: ОБУЧЕНИЕ БИНАРНОГО КЛАССИФИКАТОРА                      │
  │                                                                 │
  │  Задача: P(qty > 0) — вероятность продажи                       │
  │                                                                 │
  │  - AutoGluon TabularPredictor (binary classification)           │
  │  - Метрика: log_loss (weighted)                                 │
  │  - Sample weight: w=7 для продаж, w=1 для не-продаж             │
  │    (отражает асимметрию wMAE)                                   │
  │  - Обучение на train_split + tuning_split                       │
  │  - Ансамбль моделей: LightGBM, CatBoost, нейросети...           │
  │    WeightedEnsemble                                             │
  │  - Опционально: sample weighting для адаптации к                │
  │    distribution shift (экспоненциальное затухание)              │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  ЭТАП 4: ОБУЧЕНИЕ РЕГРЕССОРА                                    │
  │                                                                 │
  │  Задача: предсказать k = qty / 3 (т.к. qty всегда кратен 3)     │
  │                                                                 │
  │  - AutoGluon TabularPredictor (regression)                      │
  │  - Метрика: mean_absolute_error                                 │
  │  - Обучение ТОЛЬКО на записях с qty > 0 (~13% данных)           │
  │  - Обучение на train_split + tuning_split (positive-only)       │
  │  - Ансамбль моделей: LightGBM, CatBoost... WeightedEnsemble     │
  │  - Опционально: sample weighting для регрессора                 │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  ЭТАП 5: ОПТИМИЗАЦИЯ ПОРОГОВ НА HOLDOUT                         │
  │                                                                 │
  │  5a. Глобальный порог (baseline)                                │
  │      - Перебор порогов от threshold_min до threshold_max        │
  │      - Выбор порога, минимизирующего wMAE на holdout            │
  │                                                                 │
  │  5b. Группированные пороги по item_pct_nonzero                  │
  │      - Товары разбиваются на 5 групп по проценту ненулевых      │
  │        продаж в истории:                                        │
  │        * very_rare: 0-3%                                        │
  │        * rare: 3-8%                                             │
  │        * medium: 8-15%                                          │
  │        * frequent: 15-30%                                       │
  │        * very_frequent: 30-100%                                 │
  │      - Для каждой группы свой оптимальный порог                 │
  │      - Если группированные пороги хуже глобального →            │
  │        автоматический fallback на единый порог                  │
  │                                                                 │
  │  Логика применения:                                             │
  │   - Если P(sale) >= threshold_group → qty = round(k_pred) * 3   │
  │   - Иначе → qty = 0                                             │
  │   - Округление до кратного 3, клиппинг >= 0                     │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  ЭТАП 6: RETRAIN НА ПОЛНЫХ ДАННЫХ                               │
  │                                                                 │
  │  После оптимизации порогов на holdout:                          │
  │   - Классификатор переобучается на ВСЕХ train данных            │
  │   - Регрессор переобучается на ВСЕХ positive train данных       │
  │   - Используются оптимизированные пороги для финального         │
  │     submission                                                  │
  │                                                                 │
  │  Сохраняются модели:                                            │
  │   - classifier_final/ — финальный классификатор                 │
  │   - regressor_final/ — финальный регрессор                      │
  │   - config.json — конфигурация с порогами и метриками           │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  ЭТАП 7: ГЕНЕРАЦИЯ SUBMISSION                                   │
  │                                                                 │
  │  - Предсказания классификатора: P(sale)                         │
  │  - Предсказания регрессора: k → qty = k * 3                     │
  │  - Применение оптимизированных порогов (группированных или      │
  │    глобального)                                                 │
  │  - Постпроцессинг: округление до кратного 3, клиппинг >= 0      │
  │                                                                 │
  │  Выход: submission_tabular.csv                                  │
  └─────────────────────────────────────────────────────────────────┘


РЕПЛИКАЦИЯ МЕТРИКИ 0.370: replicate_370.py
------------------------------------------

Для точной репликации лучшего результата (тестовая wMAE = 0.370) используется
скрипт replicate_370.py, который:

1. Загружает обученные модели из models_tabular/:
   - classifier_final/
   - regressor_final/
   - config.json (с оптимизированными порогами)

2. Использует КОНКРЕТНЫЕ модели из ансамбля:
   - Классификатор: LightGBMLarge
   - Регрессор: WeightedEnsemble_L2

3. Применяет оптимизированные пороги из config.json по item_pct_nonzero:
  very_rare      : thr=0.18, n=4705, predicted_nonzero=174
  rare           : thr=0.67, n=2653, predicted_nonzero=174
  medium         : thr=0.24, n=2001, predicted_nonzero=446
  frequent       : thr=0.20, n=1372, predicted_nonzero=557
  very_frequent  : thr=0.14, n=2125, predicted_nonzero=1661

4. Генерирует submission с именем файла, включающим названия моделей:
   - submission_tabular_LightGBMLarge_WeightedEnsemble_L2.csv

ВАЖНО: replicate_370.py использует config.py и features_365.py
       (специальные версии конфига и фичей для репликации).


FEATURE ENGINEERING
-------------------------------

1. ИСХОДНЫЕ ПРИЗНАКИ (4):
   price, is_promo, prev_leftovers, sneakers_google_trends

2. КАЛЕНДАРНЫЕ (7):
   day_of_week, day_of_month, week_of_year, month, is_weekend,
   is_holiday (RU holidays), days_to_holiday

3. ITEM STATIC — агрегаты по истории товара (14):
   item_total_qty, item_mean_qty, item_std_qty, item_max_qty,
   item_pct_nonzero, item_mean_price, item_std_price,
   item_min_price, item_max_price, item_mean_leftovers,
   item_n_days, item_pct_promo, item_price_range_pct, item_cv_qty

4. PRICE RELATIVE — цена относительно истории товара (4):
   price_vs_item_mean, price_vs_item_min, price_vs_item_max,
   price_discount_pct

5. LEFTOVERS DERIVED (2):
   log_leftovers, leftovers_vs_item_mean

6. FORWARD/BACKWARD DELTA (4):
   fwd_delta_lo, fwd_delta_lo_raw, bwd_delta_lo

7. PRICE CHANGES (4):
   price_change_1d, price_change_7d, price_dropped, price_dropped_big

8. SELL RATE (2):
   item_sell_rate, expected_qty

9. LAG FEATURES — rolling stats (3):
   price_mean_7d, price_mean_14d, leftovers_mean_7d
   (Примечание: НЕ используются лаги по qty для избежания overfitting)

10. PROMO FEATURES (1):
    promo_days_7d

11. MARKET FEATURES (4):
    market_pct_promo, market_mean_price, market_mean_leftovers,
    price_vs_market

12. INTERACTION FEATURES (4):
    promo_and_drop, promo_x_discount, leftovers_x_promo,
    leftovers_x_pct_nz

Итого: 69 фичей


ВАЛИДАЦИЯ
---------
- Time-based split: предпоследние N дней = tuning валидация; 
  последние 14 дней train = holdout валидация (повторяет тестовый формат)
- wMAE на валидации используется для:
  - Выбора порога классификатора (глобального и per-group)
  - Сравнения grouped vs global threshold (автоматический fallback)
  - Выбора лучшей модели в AutoGluon


ВОСПРОИЗВОДИМОСТЬ
-----------------
- Глобальный seed=42 зафиксирован во всех скриптах (Python, NumPy, PyTorch)
- Seed передаётся в AutoGluon: ag_args_fit={"random_seed": 42}
- PYTHONHASHSEED зафиксирован через os.environ
- torch.backends.cudnn.deterministic = True (для GPU)
- Все параметры вынесены в единый config.py (dataclass Config)
- Seed сохраняется в models_tabular/config.json
- Примечание: полная битовая воспроизводимость при запуске train_tabular_370.py
  невозможна из-за time_limit в AutoGluon (на разном железе обучится разное число моделей),
  поэтому надо использовать replicate_370.py


БЕЙЗЛАЙНЫ (wMAE на train)
--------------------------
- Все нули:        3.336
- Глобальное среднее: 3.320
- Per-item mean:   2.468
- Per-item median: 2.637


ПОРЯДОК ЗАПУСКА
---------------

  # Шаг 1: Обучение моделей и оптимизация порогов
  python train_tabular_370.py
  
  Результат:
  - models_tabular/classifier_final/ — обученный классификатор
  - models_tabular/regressor_final/ — обученный регрессор
  - models_tabular/config.json — конфигурация с оптимизированными порогами
  - submission_tabular.csv — базовый submission

  # Шаг 2: Репликация лучшего результата (wMAE = 0.370)
  python replicate_370.py
  
  Результат:
  - submission_tabular_LightGBMLarge_WeightedEnsemble_L2.csv
    (финальный submission с метрикой 0.370 на тесте)

  Гиперпараметры настраиваются в config.py (единый конфиг).


СТРУКТУРА ПРОЕКТА
-----------------

wb_hackathon/
│
├── config.py                  # Централизованный конфиг (основные гиперпараметры)
│
├── train.parquet              # Обучающие данные (предоставлены)
├── test.parquet               # Тестовые данные (предоставлены)
├── sample_submission.csv      # Шаблон submission (предоставлен)
│
├── features_365.py            # Модуль feature engineering
│
├── train_tabular_370.py       # Основной pipeline: обучение + оптимизация порогов
├── replicate_370.py           # Репликация метрики 0.370 (использует конкретные модели)
│
├── models_tabular/            # Директория с обученными моделями
│   ├── classifier_final/      # Финальный классификатор
│   ├── regressor_final/       # Финальный регрессор
│   └── config.json            # Конфигурация с порогами и метриками
│
├── ReadME.md                  # Этот файл
