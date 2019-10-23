import numpy as np
import pandas as pd
import xgboost
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':6})

from sklearn import model_selection

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score


from xgboost import plot_importance



### get the correlation values between parameters
def correlation_plot(data):
	corr = data.corr(method='pearson')
	corr.to_csv('corr.csv')		# create csv file

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(corr, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0,corr.shape[0],1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_yticklabels(names)
	plt.savefig('corr.png')		# create image file



def time_series_validation(data):


    data_w = data.dropna()
    values = data_w.values
    X = values[:,0:-1]
    y = values[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    data_wo = data.drop(['builder_id'], axis=1)
    data_wo = data_wo.dropna()
    values = data_wo.values
    values = values[values[:,0].argsort(),:]

    X = values[:,0:-1]

    year = np.zeros(len(X), dtype='int')
    month = np.zeros(len(X), dtype='int')
    day = np.zeros(len(X), dtype='int')
    for i in range(len(X)):
        s = X[i, 0]
        year[i] = int(s[0:4])
        month[i] = int(s[5:7])
        day[i] = int(s[8:10])

    X = np.c_[year.reshape(-1,1), month.reshape(-1, 1), X[:, 1:-1]]
    y = values[:,-1]
    n = len(X)
    m = int(n/10)
    a = range(1,11)
    a = np.array(a) * m

    a[9] = n
    time_kfold_result = np.zeros(5)
    all_pred = []
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.05, gamma=0, subsample=0.6, colsample_bytree=1, max_depth=10)
    print("before fitting")
    xgb.fit(X, y)
    print("data fit done")


    year_k  = 6
    XX = np.copy(X)
    for k in range(1, year_k + 1):
        X2 = np.copy(XX)
        for i in range(len(X2)):
            if X2[i, 1] <= k:
                X2[i,0] = X2[i,0] - 1
                X2[i,1]= X2[i,1] - k + 12
            else:
                X2[i,1] -= k
        y2 = xgb.predict(X2)
        X = np.c_[y2.reshape(-1, 1), X]


    xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.06, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=10)
    for k in range(0,5):
        i = a[k + 4]
        j = a[k + 5]


        X_train = X[0:i, :]
        X_test = X[i:j,:]
        y_train = y[0:i]
        y_test = y[i:j]

        print("%dth kfold : n_train = %d , n_test = %d" %(k, len(X_train), len(X_test)))
        xgb.fit(X_train, y_train)


        predictions = xgb.predict(X_test)

        if len(all_pred) == 0:
            all_pred = predictions
        else:
            all_pred = np.r_[all_pred, predictions]
        np.save('predict%d.npy' % k, predictions)
        time_kfold_result[k] = explained_variance_score(predictions, y_test)

        plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
        plt.show()
        print("%dth kfold: %f" % (k, time_kfold_result[k]))


    plt.scatter(range(len(xgb.feature_importances_)), xgb.feature_importances_)
    plt.show()
    print("mean: ", np.mean(time_kfold_result))


### predict the price of test data
def predict(data, test_data):

    data_wo = data.drop(['builder_id'], axis=1)
    data_wo = data_wo.dropna()
    values = data_wo.values
    values = values[values[:, 0].argsort(), :]

    X = values[:, 0:-1]

    year = np.zeros(len(X), dtype='int')
    month = np.zeros(len(X), dtype='int')
    day = np.zeros(len(X), dtype='int')
    for i in range(len(X)):
        s = X[i, 0]
        year[i] = int(s[0:4])
        month[i] = int(s[5:7])
        day[i] = int(s[8:10])
    X = np.c_[year.reshape(-1, 1), month.reshape(-1, 1), X[:, 1:-1]]


    y = values[:, -1]
    test = test_data.values
    X_test = test
    year2 = np.zeros(len(X_test), dtype='int')
    month2 = np.zeros(len(X_test), dtype='int')
    day2 = np.zeros(len(X_test), dtype='int')
    for i in range(len(X_test)):
        s = X_test[i, 0]
        year2[i] = int(s[0:4])
        month2[i] = int(s[5:7])
        day2[i] = int(s[8:10])
    X_test = np.c_[year2.reshape(-1, 1), month2.reshape(-1, 1), X_test[:, 1:-1]]

    y_test = test[:,-1]
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.05, gamma=0, subsample=0.6, colsample_bytree=1,
                               max_depth=10)
    print("predict_before_fit")
    xgb.fit(X, y)
    print("predict_after_fit")
    pred = xgb.predict(X)
    pred_base = xgb.predict(X_test)
    np.savetxt('predict_base.csv', pred_base, delimiter=' ')

    year_k = 6
    XX = np.copy(X)
    for k in range(1, year_k + 1):
        X2 = np.copy(XX)
        for i in range(len(X2)):
            if X2[i, 1] <= k:
                X2[i, 0] = X2[i, 0] - 1
                X2[i, 1] = X2[i, 1] - k + 12
            else:
                X2[i, 1] -= k
        y2 = xgb.predict(X2)
        X = np.c_[y2.reshape(-1, 1), X]

    XX = np.copy(X_test)
    for k in range(1, year_k + 1):
        X2 = np.copy(XX)
        for i in range(len(X2)):
            if X2[i, 1] <= k:
                X2[i, 0] = X2[i, 0] - 1
                X2[i, 1] = X2[i, 1] - k + 12
            else:
                X2[i, 1] -= k
        y2 = xgb.predict(X2)
        X_test = np.c_[y2.reshape(-1, 1), X_test]

    xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.06, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=10)
    print("predict_before_fit2")
    xgb.fit(X, y)
    print("predict_after_fit2")
    test_predict = xgb.predict(X_test)
    np.savetxt('predict_final.csv', test_predict, delimiter=' ')
    print("predict done")


names = ['contract_date', 'latitude', 'longitude', 'altitude', '1st_region', '2nd_region', 'road_id', 'apartment_id', 'floor', 'angle', 'area', 'car#_parkinglot', 'area_parkinglot', 'external_vehical', 'management_fee', 'households#', 'age', 'builder_id', 'construction_date', 'built_year', 'school#', 'bus_station#', 'subway_station#', 'price']
train_data = pd.read_csv('data_train.csv', names=names)
print("reading data done")
#correlation_plot(train_data)
train_data = train_data.drop([ 'altitude', '1st_region','2nd_region','road_id','car#_parkinglot','area_parkinglot','households#', 'construction_date','bus_station#','external_vehical', 'subway_station#','school#'], axis=1)
#time_series_validation(train_data)
test_names = ['contract_date', 'latitude', 'longitude', 'altitude', 'apartment_id', 'floor', 'angle', 'area', 'external_vehical', 'management_fee', 'households#', 'age', 'built_year', 'school#', 'bus_station#', 'subway_station#']
test_data = pd.read_csv('data_test_no_blank_final.csv', names=test_names)
test_data = test_data.drop(['altitude','external_vehical','households#','school#', 'bus_station#', 'subway_station#'], axis=1)

predict(train_data, test_data)
