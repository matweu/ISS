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
