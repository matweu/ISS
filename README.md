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

Структура на этапе ЛР1:

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

После Feature Engineering был получен 51 признак.

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
sh start_mlflow.sh
```

После запуска интерфейс MLflow доступен по адресу:

http://localhost:5000

В нём можно просмотреть эксперименты, прогоны, метрики и зарегистрированные модели.

Модель зарегистрирована под именем `mobile-price-classifier`.
Production-версия выбирается в реестре моделей по тегу `Production=true`.
Её `run_id` используется при выгрузке модели для ЛР3.

---

# Лабораторная работа №3

## Создание сервиса предсказаний

В третьей лабораторной работе создан REST API для предсказания ценовой категории
мобильного телефона на основе Production-модели, обученной в ЛР2.

Сервис разработан на **FastAPI**, запускается сервером **Uvicorn** и
разворачивается в **Docker-контейнере**. Модель загружается из файла `model.pkl`.

На вход подаются характеристики телефона, на выходе возвращается класс
`price_range` от `0` до `3`.

## Структура сервиса

```text
ml_lab/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── clean_dataset.pkl
├── eda/
│   └── ...
├── research/
│   ├── research.ipynb
│   ├── production_features.txt
│   └── ...
├── mlflow/
│   ├── start_mlflow.sh
│   ├── mlruns.db
│   └── mlartifacts/
├── services/
│   ├── ml_service/
│   │   ├── main.py
│   │   ├── api_handler.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── models/
│       ├── get_model.py
│       └── model.pkl
├── .gitignore
├── README.md
└── requirements.txt
```

Назначение файлов:

| Файл | Назначение |
| --- | --- |
| `services/ml_service/main.py` | Создание FastAPI-приложения и обработка HTTP-запросов. |
| `services/ml_service/api_handler.py` | Класс `FastAPIHandler`, загрузка модели и вызов `predict()`. |
| `services/ml_service/requirements.txt` | Зависимости, устанавливаемые в Docker-образ. |
| `services/ml_service/Dockerfile` | Инструкции сборки образа и запуска сервиса. |
| `services/models/get_model.py` | Скачивание Production-модели из MLflow по `run_id`. |
| `services/models/model.pkl` | Обученная модель для локального инференса. |

## Подготовка окружения

Перейти в корень проекта и активировать окружение:

```bash
cd ~/ml_lab
source .venv_ml_lab/bin/activate
pip install -r requirements.txt
```

## Получение модели из MLflow

Запустить MLflow в отдельном терминале:

```bash
cd ~/ml_lab/mlflow
sh start_mlflow.sh
```

Открыть http://localhost:5000, найти зарегистрированную модель
`mobile-price-classifier`, её версию с тегом `Production=true` и скопировать `run_id`.

Указать этот идентификатор в `services/models/get_model.py`.
В другом терминале выполнить:

```bash
cd ~/ml_lab
source .venv_ml_lab/bin/activate
python services/models/get_model.py
ls -lh services/models/model.pkl
```

Файл `model.pkl` должен появиться в каталоге `services/models/`.
После выгрузки модели MLflow не требуется для обработки HTTP-запросов.

## Локальный запуск FastAPI

```bash
cd ~/ml_lab/services/ml_service
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

После запуска открывается Swagger UI: http://localhost:8000/docs.

Проверка корневого маршрута:

```bash
curl http://localhost:8000/
```

Ожидаемый ответ:

```json
{"Hello":"World"}
```

Перед запуском контейнера остановить локальный сервер сочетанием `Ctrl+C`,
чтобы освободить порт `8000`.

## Docker: сборка образа

Используется базовый образ `python:3.11-slim`.

```bash
cd ~/ml_lab/services/ml_service
docker build -t mobile-price-service:1 .
```

- `-t mobile-price-service:1` — название образа и его версия.
- `.` — контекст сборки (текущая директория).

Проверка наличия образа:

```bash
docker images
```

## Docker: запуск контейнера

Команда выполняется **из папки `services/ml_service`**:

```bash
cd ~/ml_lab/services/ml_service
docker run -d \
  --name mobile-price-service \
  -p 8000:8000 \
  -v "$(pwd)/../models:/models:ro" \
  mobile-price-service:1
```

Аргументы команды:

- `-d` — фоновый режим;
- `--name` — имя контейнера;
- `-p 8000:8000` — связь порта хоста `8000` с портом контейнера `8000`;
- `-v` — подключение директории `services/models/` к `/models` внутри контейнера;
- `:ro` — доступ к модели только для чтения.

Контейнер читает обученную модель по пути `/models/model.pkl`.
Сам файл модели не требуется включать в Docker-образ.

## Проверка контейнера

```bash
docker ps
docker logs mobile-price-service
```

Интерфейс API: http://localhost:8000/docs.

## API сервиса

### GET `/`

Проверка доступности приложения:

```bash
curl http://localhost:8000/
```

Ответ:

```json
{"Hello":"World"}
```

### POST `/api/prediction`

`item_id` передаётся как параметр URL, характеристики телефона — в JSON-теле запроса.

Пример запроса:

```bash
curl -X POST \
  "http://localhost:8000/api/prediction?item_id=123" \
  -H "Content-Type: application/json" \
  -d '{
    "battery_power": 842,
    "blue": 0,
    "clock_speed": 2.2,
    "dual_sim": 0,
    "fc": 1,
    "four_g": 0,
    "int_memory": 7,
    "m_dep": 0.6,
    "mobile_wt": 188,
    "n_cores": 2,
    "pc": 2,
    "px_height": 20,
    "px_width": 756,
    "ram": 2549,
    "sc_h": 9,
    "sc_w": 7,
    "talk_time": 19,
    "three_g": 0,
    "touch_screen": 0,
    "wifi": 1,
    "pixel_area": 15120
  }'
```

Список входных признаков должен соответствовать файлу
`research/production_features.txt` из ЛР2.

Пример структуры ответа (конкретный класс зависит от модели):

```json
{
  "item_id": 123,
  "predict": 1
}
```

`predict` — предсказанная ценовая категория: 0, 1, 2 или 3.
Преобразования признаков выполняются обученным Pipeline из ЛР2.

## Управление контейнером

```bash
# Показать запущенные контейнеры
docker ps

# Просмотреть логи
docker logs mobile-price-service

# Остановить контейнер
docker stop mobile-price-service

# Запустить существующий контейнер
docker start mobile-price-service

# Перезапустить контейнер
docker restart mobile-price-service

# Удалить контейнер перед повторным docker run
docker rm -f mobile-price-service
```

Для пересборки после изменения исходного кода:

```bash
cd ~/ml_lab/services/ml_service
docker rm -f mobile-price-service
docker build -t mobile-price-service:1 .
docker run -d --name mobile-price-service -p 8000:8000 -v "$(pwd)/../models:/models:ro" mobile-price-service:1
```

## Результат ЛР3

Подготовлен сервис, который принимает характеристики мобильного телефона
через HTTP POST, выполняет предсказание сохранённой ML-моделью и возвращает
JSON с идентификатором объекта и его ценовой категорией.

Сервис можно запускать как локально с Uvicorn, так и в Docker-контейнере.
