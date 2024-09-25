import os
import sys

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from src.utils import evaluate_models

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts/pickle', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gredient Boosting": GradientBoostingRegressor(),
                "Linear Regressor": LinearRegression(),
                "K-Neibours Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Adaboost": AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test,
                                                models=models)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            ## Grid Search for Random Forest and XGB Regressor
            if best_model_name == "Random Forest":
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15]
                }
                grid_search = GridSearchCV(best_model, param_grid, cv=5)
                grid_search.fit(X_train, y_train)
                best_model = best_model.set_params(**grid_search.best_params_)
            elif best_model_name == "XGB Regressor":
                param_grid = {
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7]
                }
                grid_search = GridSearchCV(best_model, param_grid, cv=5)
                grid_search.fit(X_train, y_train)
                best_model = best_model.set_params(**grid_search.best_params_)

            ## Random Search for Gradient Boosting
            elif best_model_name == "Gredient Boosting":
                param_dist = {
                    'learning_rate': uniform(0.01, 0.3),
                    'max_depth': [3, 5, 7, 10]
                }
                random_search = RandomizedSearchCV(best_model, param_dist, n_iter=20, cv=5)
                random_search.fit(X_train, y_train)
                best_model = best_model.set_params(**random_search.best_params_)

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found on both training and testing dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            # Print the R² score to the terminal
            print(f'R² Score of the best model ({best_model_name}): {r2_square:.4f}')

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)