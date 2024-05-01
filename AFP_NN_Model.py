# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 16:33:41 2021

@author: saxen
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
from sklearn import model_selection
from sklearn.metrics import r2_score
import glob
import os
import gc
from joblib import Parallel, delayed

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





# Split features and target
x = train.drop(['post_money_valuation'], axis = 1)
y = train[['post_money_valuation']]
x_test = test.drop(['post_money_valuation'], axis = 1)



### Scaling ####
scale_x = StandardScaler()
scale_y = StandardScaler()

X_train = scale_x.fit_transform(x)
y_train = scale_y.fit_transform(y).reshape(-1,1)
X_test = scale_x.fit_transform(x_test)


nrow= X_train.shape[1:]




### Model ###

model= Sequential()

model.add(Dense(100 , input_shape=(nrow), activation='relu', kernel_initializer='normal'))

model.add(Dense(75 , activation='relu', kernel_initializer='normal' ))


model.add(Dense(25 , activation='relu', kernel_initializer='normal' ))



model.add(Dense(1,kernel_initializer='normal',activation='linear'))



epochs = 50
learning_rate = 0.5
decay_rate =learning_rate/epochs
adam = Adam(lr = learning_rate, decay= decay_rate)


model.compile(optimizer = 'adam', loss= 'mse')

model.fit(X_train,y_train,epochs=50,validation_split=0.3)

y_pred = model.predict(X_test)
y_pred = scale_y.inverse_transform(y_pred)
y_test = test['post_money_valuation']
#rmspe = (np.sqrt(np.mean(np.square((y_train-y_pred)/y_train))))
#print(rmspe)

result = pd.DataFrame(y_pred,y_test)
print(result)
print(y_pred)
print(y_test)


