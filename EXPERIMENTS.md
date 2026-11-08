# Результаты экспериментов

Главная метрика проекта — ROC-AUC. Ниже результаты на одной отложенной выборке: 20% обучающих данных, `random_state=42`. Эти числа нельзя напрямую сравнивать с результатами кросс-валидации.

| Модель и признаки | ROC-AUC | PR-AUC | Accuracy | Запуск MLflow |
| --- | ---: | ---: | ---: | --- |
| LogisticRegression, baseline | 0.77199 | 0.25893 | 0.91956 | [открыть](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/ed4d9b9ebcd3427b9cadaf4ee65eb089) |
| RandomForest, baseline | 0.75817 | 0.24687 | 0.91927 | [открыть](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/fe495fb84d9d497bb911f86e5a846f84) |
| CatBoost, baseline | 0.77978 | 0.27641 | 0.92007 | [открыть](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/9bea9c4a1f2b4c0abc90585d7ab99d08) |
| LightGBM, baseline | 0.78297 | 0.27954 | 0.92010 | [запуск 1](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/3a579593036a43178ce7337cbdc7f7d6), [запуск 2](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/98429690ff014b158f2ab9a35e711755) |
| LightGBM, + `credit_to_annuity` | 0.78543 | 0.28238 | 0.91999 | [открыть](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/c4b7f173c050489495cca239fa625be6) |
| LightGBM, + сумма долга и `CREDIT_DEBT_RATIO` | 0.78638 | 0.28447 | 0.92041 | [открыть](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/a0199410be9d41b7b850d762ae6351b9) |
| LightGBM, + `credit_goods_price_ratio`, `credit_downpayment`, `AGE_INT` | 0.78775 | 0.28769 | 0.92033 | [запуск 1](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/14873a10cfe74027b0df7c1a6c697058), [запуск 2](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/49b91c31cf2b4a2ca9104438c54fb344) |
| CatBoost, тот же набор признаков | 0.78368 | 0.28259 | 0.91992 | [запуск 1](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/adff0421ef714c64a10a551833f5ad65), [запуск 2](https://dagshub.com/Mu2Ka/Credit-Default-Risk-mu2ka.mlflow/#/experiments/0/runs/774b1130a31b4a279383980c2f80d5f9) |

Новые признаки добавлялись последовательно. Поэтому строка с долгом включает `credit_to_annuity`, а строки с тремя последними признаками включают и долг, и `credit_to_annuity`. Прирост последнего запуска нельзя приписать каждому из трёх признаков отдельно.

Следующий шаг — сравнить варианты на одинаковых fold кросс-валидации. Для каждого варианта записывать в MLflow средний ROC-AUC, разброс и оценки отдельных fold. Выбирать модель по CV, а отложенную выборку использовать для проверки и анализа ошибок.

## Как записывать новые результаты

- Для обучения указывать название набора признаков: `train_model(name="LightGBM", name_experiment="имя_набора")`.
- Для кросс-валидации указывать то же название в `compare_models(..., feature_set="имя_набора")`. Функция сохранит JSON и отдельный запуск MLflow для каждой модели.
- Все 11 старых запусков уже подписаны тегами в DagsHub. Скрипт `python src/label_existing_runs.py` можно запустить повторно при необходимости; он не переобучает модели и не меняет их метрики.
- Когда выбрана лучшая модель, открыть её запуск в MLflow и зарегистрировать модель оттуда. Промежуточные варианты теперь сохраняются как артефакты запусков без автоматической регистрации.
