# Mobile Price Classification

Лабораторная работа №1 — настройка окружения и разведочный анализ данных.

## Описание проекта

В проекте проводится разведочный анализ данных датасета **Mobile Price Classification**.

Задача — исследовать характеристики мобильных телефонов и подготовить данные
для дальнейшего построения модели классификации ценового диапазона телефона.

Целевая переменная:

`price_range`

Она принимает четыре значения:

- `0` — низкая ценовая категория
- `1` — средняя ценовая категория
- `2` — высокая ценовая категория
- `3` — очень высокая ценовая категория

## Датасет

Используется датасет:

**Mobile Price Classification**

Источник:

https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

Для анализа используется файл:

`data/train.csv`

Файлы с данными не сохраняются в Git-репозитории.

## Структура проекта

```text
ml_lab/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── clean_dataset.pkl
│
├── eda/
│   ├── eda.ipynb
│   ├── graph1_price_range.png
│   ├── graph2_ram_price.png
│   ├── graph3_battery_price.png
│   ├── graph4_correlation.png
│   ├── graph5_wifi.png
│   ├── graph6_interactive.html
│   └── graph7_pixel_area.png
│
├── .gitignore
├── README.md
└── requirements.txt
```

# Лабораторная работа №2
 
## Проведение экспериментов по настройке модели
 
Во второй лабораторной работе проведены эксперименты по настройке модели
для задачи классификации ценового диапазона мобильного телефона.
 
В качестве основной модели используется `RandomForestClassifier`.
 
Для отслеживания экспериментов используется MLflow.
 
## Подготовка данных
 
Для экспериментов используется очищенный датасет, полученный в результате
выполнения лабораторной работы №1:
 
`data/clean_dataset.pkl`
 
Целевая переменная:
 
`price_range`
 
Данные разделены на обучающую и тестовую выборки в соотношении:
 
- train — 75%
- test — 25%
 
Для сохранения распределения классов используется стратифицированное
разделение.
 
## Pipeline baseline-модели
 
Baseline pipeline состоит из двух этапов:
 
1. Обработка признаков.
2. Обучение модели.
 
Для числовых признаков используется:
 
`StandardScaler`
 
Для категориальных признаков используется:
 
`TargetEncoder`
 
В качестве модели используется:
 
`RandomForestClassifier`
 
Для оценки качества используются метрики:
 
- Precision
- Recall
- F1-score
- ROC AUC
 
## Feature Engineering
 
Для расширения пространства признаков использовались средства `scikit-learn`.
 
Применялись:
 
- `PolynomialFeatures` с `degree=2`;
- `KBinsDiscretizer`;
- `StandardScaler`;
- `TargetEncoder`.
 
Для PolynomialFeatures и KBinsDiscretizer использовались признаки:
 
- `ram`
- `battery_power`
- `pixel_area`
 
До преобразований использовался 21 признак.
 
После Feature Engineering было получено 51 признаков.
 
Названия сформированных признаков сохранены в файле:
 
`research/sklearn_feature_names.txt`
 
## Feature Selection
 
После Feature Engineering был выполнен отбор наиболее значимых признаков.
 
При выборе использовались результаты разведочного анализа данных,
полученные в лабораторной работе №1.
 
Основное внимание уделялось признакам, связанным с:
 
- `ram`
- `battery_power`
- `pixel_area`
- `px_height`
- `px_width`
 
Индексы выбранных признаков сохранены в:
 
`research/selected_indices.txt`
 
Названия выбранных признаков сохранены в:
 
`research/selected_features.txt`
 
## Подбор гиперпараметров
 
Для настройки RandomForest используется библиотека `Optuna`.
 
Подбираются следующие параметры:
 
- `n_estimators`
- `max_depth`
- `max_features`
 
Проведено не менее 10 trials.
 
В качестве оптимизируемой метрики используется `F1-score`.
 
Так как большее значение F1 соответствует лучшему качеству модели,
направление оптимизации задано как:
 
`maximize`
 
Результаты Optuna сохранены в:
 
`research/optuna_trials.txt`
 
## MLflow
 
MLflow запускается локально.
 
В качестве backend store используется SQLite.
 
Для запуска MLflow необходимо перейти в директорию:
 
```bash
cd mlflow
 
 
--


