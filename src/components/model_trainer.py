#!/usr/bin/env python
# coding: utf-8

# Import Required Libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import *
from dataclasses import dataclass

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# Set the figure size for better visualization
plt.figure(figsize=(15, 8))

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

# Modelling
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge,Lasso
import xgboost as xgb
import lightgbm as lgb


# Initialize Model Trainer Configuration

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    model_report_file_path = os.path.join('artifacts','model_report.csv')
    


# # Step 3: Model Development

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train, test):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            
            # Define Variables
            target_column_name = 'total_rooms_sold_lag_3'
            drop_columns = [target_column_name]

            X_train = train.drop(columns=drop_columns,axis=1)
            y_train=train[target_column_name]

            X_test =test.drop(columns=drop_columns,axis=1)
            y_test =test[target_column_name]
            
         
            # Define the models and their hyperparameters
            
            lr = LinearRegression()
            ridge = Ridge()
            lasso = Lasso()
            XGB = xgb.XGBRegressor()
            LGM = lgb.LGBMRegressor()
            
            models = {
                'Linear Regression': {
                    'model': lr,
                    
                    'params': {
                        'normalize': [True, False],
                        'fit_intercept': [True, False]
                    }
                },
                'Ridge Regression': {
                    'model': ridge,
                    'params': {
                        'alpha': np.logspace(-4, 2, 100),
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                    } 
                },
                'Lasso Regression': {
                    'model': lasso,
                    'params': {
                         'alpha': [0.001, 0.01, 0.1, 1, 10, 100]  # Values for the regularization strength
                    }
                },
                'XGBoost': {
                    'model': XGB,
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'num_leaves': [10,100],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'gamma': [0, 1, 5]
                    }
                },
                'LightGBM': {
                    'model': LGM,
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'num_leaves': [10,100],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'reg_alpha': [0, 0.1, 0.5],
                        'reg_lambda': [0, 0.1, 0.5]
                    }
                }
            }

            model_report = evaluate_models(X_train,y_train,X_test,y_test,models)
            
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            # To get best model score 
            best_model_scores = min(model_report, key = lambda x: int(x[3]))

            best_model = best_model_scores[0]
            best_model_score = best_model_scores[-1] 

            '''if best_model_score < 0.6 :
                logging.info('Best model has _ Score less than 60%')
                raise CustomException('No Best Model Found')'''
            
            print(f'Best Model Found , Model Name : {best_model} , RMSE : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model} , RMSE : {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info('Model pickle file saved')
            
            
            # Model Evaluation
            ytest_pred = best_model.predict(X_test)

            MAE, MSE, RMSE  = model_metrics(y_test, ytest_pred)
            logging.info(f'Test Mean Absolute Error : {MAE}')
            logging.info('Training Mean Squared Error:', MSE)
            logging.info(f'Test Root Mean Squared Error : {RMSE}')
            
            sns.distplot(y_test - ytest_pred)
         
            logging.info('Final Model Training Completed')
            
            return MAE, MSE, RMSE
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

