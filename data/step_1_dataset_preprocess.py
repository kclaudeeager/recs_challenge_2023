import numpy as np 
import pandas as pd 
import os 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import lightgbm as lgb

# Load the dataset
data_files = []
for i in range(30):
    file_path = "/kaggle/input/train-recsys-challenge-2023/{:012d}.csv".format(i)
    data_files.append(file_path)

dfTrain = []
for file_path in data_files:
    dfTrain.append(pd.read_csv(file_path, delimiter="\t"))

train_data = pd.concat(dfTrain, axis=0)
train_data = train_data.reset_index(drop=True)

test_size = 0.2

# Separate target variables
target = train_data[['is_clicked', 'is_installed']]
features = train_data.drop(['is_clicked', 'is_installed'], axis=1)

train_data, test_data, train_labels, test_labels = train_test_split(features, target, test_size=test_size)

print("Data Loaded")

# Preprocess the data
test_data['is_clicked'] = -1
test_data['is_installed'] = -1

train_data['x_1'] = train_data['f_1'] % 7
test_data['x_1'] = test_data['f_1'] % 7

train_data_full = train_data.copy()
valid_data = train_data[train_data['f_1'] == 66]
train_data = train_data[train_data['f_1'] < 66]

y_valid = pd.DataFrame(columns=['f_30', 'f_31'], index=valid_data.index)
y_test = pd.DataFrame(columns=['f_30', 'f_31'], index=test_data.index)

train_na = train_data[train_data['f_30'].isna()]
valid_na = valid_data[valid_data['f_30'].isna()]
test_na = test_data[test_data['f_30'].isna()]

train_not_na = train_data[~train_data['f_30'].isna()]
X_train = train_not_na.drop(['f_30', 'f_31'], axis=1)
y_train = train_not_na[['f_30', 'f_31']]

X_train_na = train_na.drop(['is_clicked', 'is_installed'], axis=1)
X_valid_na = valid_na.drop(['is_clicked', 'is_installed'], axis=1)
X_test_na = test_na.drop(['is_clicked', 'is_installed'], axis=1)

gbm1 = lgb.LGBMClassifier(objective='binary',
                          metric='auc',
                          random_state=42,
                          learning_rate=0.05,
                          max_depth=3,
                          num_leaves=7, verbose=3)
gbm1.fit(X_train, y_train.f_30)
X_train_na['f_30'] = gbm1.predict(X_train_na)
X_valid_na['f_30'] = gbm1.predict(X_valid_na)
X_test_na['f_30'] = gbm1.predict(X_test_na)

gbm2 = lgb.LGBMClassifier(objective='binary',
                          metric='auc',
                          random_state=42,
                          learning_rate=0.05,
                          max_depth=3,
                          num_leaves=7, verbose=3)
gbm2.fit(X_train, y_train.f_31)

X_train_na['f_31'] = gbm2.predict(X_train_na.drop(['f_30'], axis=1))
X_valid_na['f_31'] = gbm2.predict(X_valid_na.drop(['f_30'], axis=1))
X_test_na['f_31'] = gbm2.predict(X_test_na.drop(['f_30'], axis=1))

cnt_na = np.sum(train_data.isna())
cols = cnt_na[cnt_na != 0].index
fillna_dict = dict()
for c in cols:
    if c in ['f_30', 'f_31']:
        continue
    else:
        fillna_dict[c] = np.mean(train_data[c])

if 'f_30' in X_train_na.columns and 'f_31' in X_train_na.columns:
    fill_train = X_train_na[['f_30', 'f_31']]
    fill_valid = X_valid_na[['f_30', 'f_31']]
    fill_test = X_test_na[['f_30', 'f_31']]
else:
    fill_train = pd.DataFrame()
    fill_valid = pd.DataFrame()
    fill_test = pd.DataFrame()

if 'f_30' in test_data.columns and 'f_31' in test_data.columns:
    test_data = test_data.fillna(fill_test)
    valid_data = valid_data.fillna(fill_valid)
    train_data = train_data.fillna(fill_train)

train_data = train_data.fillna(fillna_dict)
valid_data = valid_data.fillna(fillna_dict)
test_data = test_data.fillna(fillna_dict)

mms = MinMaxScaler(feature_range=(0, 1))
dense_feature = []
for i in range(42, 80):
    dense_feature.append('f_{}'.format(i))
train_data[dense_feature] = mms.fit_transform(train_data[dense_feature])
valid_data[dense_feature] = mms.transform(valid_data[dense_feature])
test_data[dense_feature] = mms.transform(test_data[dense_feature])

train_data_full = pd.concat([train_data, valid_data], axis=0).reset_index(drop=True)

# Save data
dataset_name = 'final'
if not os.path.exists("{}".format(dataset_name)):
    os.mkdir("{}".format(dataset_name))
train_data.to_csv('./{}/train_data_{}.csv'.format(dataset_name, dataset_name), index=None)
test_data.to_csv('./{}/test_data_{}.csv'.format(dataset_name, dataset_name), index=None)
valid_data.to_csv('./{}/valid_data_{}.csv'.format(dataset_name, dataset_name), index=None)
train_data_full.to_csv('./{}/train_data_{}_full.csv'.format(dataset_name, dataset_name), index=None)