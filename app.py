import hydra
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from hydra.utils import instantiate
import logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Загружаем данные
    df = pd.read_csv('train.csv')
    X = df[cfg['dataset']['columns']]
    y = df[cfg['dataset']['target']]

    # Формируем трейн/тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    # Обучение модели
    #model = LGBMRegressor()
    #lgbm_params = OmegaConf.to_container(cfg['params'])
    #model.set_params(**lgbm_params)
    model = instantiate(cfg['model'])
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred = model.predict(X_test)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    log.info(f'MSE: {mse}')

if __name__ == "__main__":
    my_app()