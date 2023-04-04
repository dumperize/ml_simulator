import pandas as pd
import numpy as np
import optuna
from solution import bias
from solution import GroupTimeSeriesSplit
from solution import mape
from solution import smape
from solution import wape
from solution import best_model
from sklearn.ensemble import GradientBoostingRegressor


def main():
    # Data loading
    df_path = "../datasets/data_train_sql.csv"
    df = pd.read_csv(df_path, parse_dates=["monday"])

    y = df.pop("y")

    # monday or product_name as a groups for validation?
    # df.drop("product_name" / "monday", axis=1, inplace=True)
    # groups = df.pop("monday" / "product_name")
    df.drop("product_name", axis=1, inplace=True)
    groups = df.pop("monday")

    X = df


    # Validation loop
    
        
    def objective(trial):
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'random_state': trial.suggest_int('random_state', 1, 1000),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1300),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'max_depth': trial.suggest_int('max_depth', 1, 10),

            # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            # 'gamma': trial.suggest_float('gamma', 0.01, 1.5),
            # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            # 'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0),
            # 'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        }
        model = GradientBoostingRegressor(**param)

        cv = GroupTimeSeriesSplit(
            n_splits=5,
            max_train_size=None,
            test_size=None,
            gap=2,
        )

        res = []
        for train_idx, test_idx in cv.split(X, y, groups):
            # Split train/test
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Fit model
            model = best_model()
            model.fit(X_train, y_train)
            # # Predict and print metrics
            y_pred = model.predict(X_test)
            est = wape(y_test, y_pred)
            res.append(est)
        return np.average(est)
    
    study = optuna.create_study(direction='minimize', study_name='regression')
    study.optimize(objective, n_trials=100)

    print('Best parameters', study.best_params)

if __name__ == "__main__":
    main()
