import pandas as pd
import numpy as np
import xgboost

from sklearn.metrics import explained_variance_score

def date_to_float(data):
	first_date = pd.Timestamp(year=1980, month=1, day=1)
	data['contract_date'] = pd.to_datetime(data['contract_date'])
	data['contract_date'] = (data['contract_date'] - first_date)/np.timedelta64(1,'D')/4748.0
	return data
c = ["contract_date", "latitude", "longitude", "altitude", 
           "1st", "2nd", "road_id", 
           "apartment_id", "floor", "angle", "area", "limit_car", 
           "area_of_parking_lot", "external_vehicle", "average_fee",
           "num_households",
           "average_age","builder_id",
           "construction_date", "built_year","num_schools",
           "num_bus","num_subway", "price"]
c_2 = ["contract_date", "latitude", "longitude", "altitude", 
           "1st", "2nd", "road_id", 
           "apartment_id", "floor", "angle", "area", "limit_car", 
           "area_of_parking_lot", "external_vehicle", "average_fee",
           "num_households",
           "average_age","builder_id",
           "construction_date", "built_year","num_schools",
           "num_bus","num_subway"]
input_df = pd.read_csv('data_train.csv', names= c)

#dataset = input_df.values
#input_df.contract_date = input_df.contract_date.str[:4].astype(int)
final = input_df.drop(["1st", "2nd", "road_id", "limit_car", "area_of_parking_lot", "construction_date", 
                       "price", "builder_id"], axis = 1)
latitude_average = np.average
final_input = final.copy()
input_lat_mean=final_input["latitude"].mean(skipna=True)
input_long_mean=final_input["longitude"].mean(skipna=True)

final_input[["latitude", "longitude"]] = final_input[["latitude", "longitude"]].fillna(value = {"latitude": input_lat_mean, "longitude":input_long_mean})
fianl_input = date_to_float(final_input)


#find columns that has missing values
missing_columns = []
for col in final_input.columns:
    is_nan_col = np.isnan(final_input[col]).values
    unique, counts = np.unique(is_nan_col, return_counts = True)
    dic = dict(zip(unique, counts))
    if True in dic.keys():
        missing_columns.append(col)



test_df = pd.read_csv('data_test.csv', names= c_2)
test_final = test_df.drop(["1st", "2nd", "road_id", "limit_car", "area_of_parking_lot", "construction_date","builder_id"]
                         , axis = 1)
test_lat_mean=test_final["latitude"].mean(skipna=True)
test_long_mean=test_final["longitude"].mean(skipna=True)

test_final[["latitude", "longitude"]] = test_final[["latitude", "longitude"]].fillna(value = {"latitude": input_lat_mean, "longitude":input_long_mean})
test_no_nan = test_final.dropna()
test_final = date_to_float(test_final)

no_nan = final_input.dropna()
for col in missing_columns:
    total_columns = final_input.columns.tolist()
    total_columns.remove(col)
    X_train = no_nan[total_columns].values
    y_train = no_nan[col].values
    X_test = test_final[test_final[col].isnull()]
    X_test = X_test.drop(col, axis = 1).values
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    indexes = np.where(np.isnan(test_final[col]))[0]
    for i in range(len(indexes)):
        value = y_pred[i]
        index = indexes[i]
        test_final.set_value(index, col, value)
test_final.to_csv("data_test_no_blank.csv")
