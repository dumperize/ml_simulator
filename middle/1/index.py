import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn import preprocessing
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error


def prepare_dataset(df):
    # преобразуем дату
    df['year'] = pd.DatetimeIndex(df['timestamp']).year
    df['month'] = pd.DatetimeIndex(df['timestamp']).month
    df['timestamp'] = pd.DatetimeIndex(df['timestamp']).values.astype(np.int64)

    # поправим явные косяки 
    df['full_sq'] = np.where(df['full_sq'] == 0, np.nan, df['full_sq']) # нулевая площадь
    df['life_sq'] = np.where(df['life_sq'] == 0, np.nan, df['life_sq']) # нулевая площадь
    df['num_room'] = np.where(df['num_room'] == 0, np.nan, df['num_room']) # нулевая количество комнат
    df['build_year'] = np.where(df['build_year'] < 1930, np.nan, df['build_year']) # ошибка в годе постройки

    # добавим фичи
    df['d_kit'] = df['kitch_sq'] / df['full_sq'] 
    df['d_lif'] = df['life_sq'] / df['full_sq']

    df['last_f'] = (df['floor'] == df['max_floor'] ).astype(bool) & df['floor'].notnull()
    df['first_f'] = (df['floor'] == 1 ).astype(int)
    df['newbie'] = (df['year'] == df['build_year']).astype(int)
    
    # укажем категориальные фичи
    df['apartment condition'] = df['apartment condition'].astype("category")
    df['sub_area'] =  df['sub_area'].fillna(10000)
    df['sub_area'] = df['sub_area'].astype("category")
    df['material'] = df['material'].astype("category")

    # заполним пропуски ближайшими соседяки
    imp_mean = KNNImputer(n_neighbors=3, weights="uniform")
    imp_mean.fit(df)
    df = pd.DataFrame(imp_mean.transform(df), index=df.index)

    return df
    
# готовим трейн
train_main = pd.read_csv('HW_train_main_data.csv', index_col='id')
train_add = pd.read_csv('HW_train_additional_data.csv',  index_col='id')
train_X = train_main.drop(columns=['price']).join(train_add)
train_X = prepare_dataset(train_X)
train_y = train_main['price']

# готовим тест
test_main = pd.read_csv('HW_test_main_data.csv',  index_col='id')
test_add = pd.read_csv('HW_test_additional_data.csv',  index_col='id')
val_X = test_main.join(test_add)
val_X = prepare_dataset(val_X)

# делим датасет
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=1)

# готовим функцию для оптуны
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.5),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'random_state': trial.suggest_int('random_state', 1, 1000)
    }
    model = xgb.XGBRegressor(**param, enable_categorical=True, tree_method='approx')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)

# подбираем параметры
study = optuna.create_study(direction='minimize', study_name='regression')
study.optimize(objective, n_trials=100)

print('Best parameters', study.best_params)


# обучаем модель с лучшими параметрами
model = xgb.XGBRegressor(**study.best_params)
model.fit(train_X, train_y)

# готовим предикшн сохраняем в файл
prediction = model.predict(val_X)
pd.DataFrame(prediction, index=val_X.index, columns=['predicted_price']).to_csv('prediction.csv')
