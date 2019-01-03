# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:22:12 2018

@author: gzy10
"""

import pandas as pd
trainx = pd.read_csv('trainx.csv')
trainy = pd.read_csv('trainy.csv')
test = pd.read_csv('testx.csv')
trainx.fillna(1,inplace=True)
test.fillna(1,inplace=True)
selected_features = ['age','job','marital','education','default','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
train_x = trainx[selected_features]
train_y = trainy.y
test_x = test[selected_features]
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
train_x = dict_vec.fit_transform(train_x.to_dict(orient='record'))
test_x = dict_vec.transform(test_x.to_dict(orient='record'))
from sklearn.ensemble import GradientBoostingRegressor
rfr = GradientBoostingRegressor()
rfr.fit(train_x, train_y)
rfr_y_predict = rfr.predict(test_x)
j=0
for i in rfr_y_predict:
    if i<=0:
    	rfr_y_predict[j] = 0
    else:
    	rfr_y_predict[j] = 1
    print(rfr_y_predict[j])
    j=j+1
data1 = pd.DataFrame(rfr_y_predict)
data1.to_csv('data1.csv')