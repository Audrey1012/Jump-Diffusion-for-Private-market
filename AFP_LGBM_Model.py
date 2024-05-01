# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 21:20:59 2021

@author: saxen
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MinMaxScaler
import xgboost
import lightgbm as lgb



train_ = pd.read_csv('C:/Users/saxen/Downloads/AFP_Data_train.csv')
b = train_[["id","company_id", "date", "status", "deal_type"]]
train_ = train_.drop(["id","company_id", "date", "status", "deal_type"],axis = 1)
b = b.apply(lambda col: pd.factorize(col, sort=True)[0])
train = pd.concat([train_, b], sort=False,axis=1)


test_ = pd.read_csv('C:/Users/saxen/Downloads/AFP_Data_test.csv')
a = test_[['id',"company_id", "date", "status", "deal_type"]]
test_ = test_.drop(["id","company_id", "date", "status", "deal_type"],axis = 1)
a = a.apply(lambda col: pd.factorize(col, sort=True)[0])
test = pd.concat([test_, a], sort=False,axis=1)



# Function to calculate the root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

# Function to early stop with root mean squared percentage error
def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

def train_and_evaluate(train, test):
    # Hyperparammeters (just basic)
    seed0=2021
    params = {
        'objective': 'rmse',
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'max_bin':100,
        'min_data_in_leaf':500,
        'learning_rate': 0.05,
        'subsample': 0.72,
        'subsample_freq': 4,
        'feature_fraction': 0.5,
        'lambda_l1': 0.5,
        'lambda_l2': 1.0,
        'categorical_column':[0],
        'seed':seed0,
        'feature_fraction_seed': seed0,
        'bagging_seed': seed0,
        'drop_seed': seed0,
        'data_random_seed': seed0,
        'n_jobs':-1,
        'verbose': -1}
    
    
    
    # Split features and target
    x = train.drop(['post_money_valuation'], axis = 1)
    y = train['post_money_valuation']
    x_test = test.drop(['post_money_valuation'], axis = 1)
    
    
    # Create out of folds array
    oof_predictions = np.zeros(x.shape[0])
    # Create test array to store predictions
    test_predictions = np.zeros(x_test.shape[0])
    # Create a KFold object
    kfold = KFold(n_splits = 5, random_state = 66, shuffle = True)
    # Iterate through each fold
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train, y_train, weight = train_weights)
        val_dataset = lgb.Dataset(x_val, y_val, weight = val_weights)
        model = lgb.train(params = params, 
                          train_set = train_dataset, 
                          valid_sets = [train_dataset, val_dataset], 
                          num_boost_round = 10000, 
                          early_stopping_rounds = 50, 
                          verbose_eval = 50,
                          feval = feval_rmspe)
        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict(x_val)
        # Predict the test set
        test_predictions += model.predict(x_test) / 5
        
    rmspe_score = rmspe(y, oof_predictions)
    print(f'Our out of folds RMSPE is {rmspe_score}')
    # Return test predictions
    return test_predictions

y_pred = train_and_evaluate(train,test)
test['post_money_valuation'] = pd.Series(y_pred)
sub = test[['post_money_valuation']]
sub.to_csv("C:/Users/saxen/Downloads/submission_.csv")


