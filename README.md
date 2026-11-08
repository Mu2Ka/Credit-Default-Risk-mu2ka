# Credit Default Risk

## Датасет

[Home Credit Default Risk на Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data). Скачайте CSV-файлы и положите их в `data/raw/`.

## Структура файлов

```text
data/raw/                  исходные CSV-файлы
data/processed/            подготовленные train и test
src/prepare_data.py        подготовка данных
src/features.py            создание признаков из дополнительных таблиц
src/preprocessing.py       заполнение пропусков и кодирование категорий
src/train_model.py         обучение моделей и кросс-валидация
src/run_experiment.py      запуск обучения LightGBM
configs/params.yaml        параметры обучения
notebooks/                 анализ ошибок и архивный ноутбук
models/                    сохранённые модели
tests/                     проверки кода
dvc.yaml                   стадии подготовки данных и обучения
EXPERIMENTS.md             таблица результатов
```

## Как запустить

Нужен Python 3.12. Выполняйте команды из корня репозитория:

```powershell
python -m pip install -r requirements.txt
python src/prepare_data.py
python src/run_experiment.py
```

Последняя команда обучает LightGBM и записывает результат в MLflow на DagsHub; для этого нужна авторизация в DagsHub. Для кросс-валидации всех моделей запустите `python src/train_model.py`.

Вместо двух последних команд можно запустить весь пайплайн через DVC:

```powershell
dvc repro
```

Результаты экспериментов: [MLflow на DagsHub](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0). Краткая таблица — в [EXPERIMENTS.md](EXPERIMENTS.md).
