import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
            try:
                logging.info("Spliting testing and training data")
                X_train,y_train,X_test,y_test=(
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )
                models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'XGB Regressor': XGBRegressor(),
                'catboost Regressor': CatBoostRegressor(verbose=0),
                'AdaBoost Regressor': AdaBoostRegressor()
                } 

                params= {
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighbors Regressor":{},              
                "Linear Regression":{},
                "XGB Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "catboost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }


                model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

                best_model_name= max(model_report, key=model_report.get)

                best_model_score= model_report[best_model_name]

                best_model= models[best_model_name]

                if best_model_score<0.6:
                    raise CustomException("No Best model found")
                logging.info(f"Best found model on both training and testing dataset")
                
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                predicted=best_model.predict(X_test)

                final_score = r2_score(y_test, predicted)
                return final_score, best_model_name, best_model   

            except Exception as e:
                raise CustomException (e, sys)
            
