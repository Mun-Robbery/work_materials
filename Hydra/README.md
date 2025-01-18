Запуск:
python app.py

Переопределение параметров:
python app.py params.max_depth=11

Задать датасет:
python app.py dataset=ds1

Multirun:
python app.py -m dataset=ds1,ds2
python app.py -m dataset=ds1,ds2 model=catboost,lgbm
