import numpy as np 
import pandas as pd 
import os 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

# Define features to use for training (excluding 'f_30' and 'f_31')
feature_columns = [col for col in train_data.columns if col not in ['f_30', 'f_31']]

X_train = train_not_na[feature_columns]
y_train = train_not_na[['f_30', 'f_31']]

X_train_na = train_na[feature_columns]
X_valid_na = valid_na[feature_columns]
X_test_na = test_na[feature_columns]

gbm1 = lgb.LGBMClassifier(objective='binary',
                          metric='auc',
                          random_state=42,
                          learning_rate=0.05,
                          max_depth=3,
                          num_leaves=7, verbose=3)
gbm1.fit(X_train, y_train.f_30)

# Fill NaN values in f_30 using predictions
train_data.loc[train_data['f_30'].isna(), 'f_30'] = gbm1.predict(X_train_na)
valid_data.loc[valid_data['f_30'].isna(), 'f_30'] = gbm1.predict(X_valid_na)
test_data.loc[test_data['f_30'].isna(), 'f_30'] = gbm1.predict(X_test_na)

gbm2 = lgb.LGBMClassifier(objective='binary',
                          metric='auc',
                          random_state=42,
                          learning_rate=0.05,
                          max_depth=3,
                          num_leaves=7, verbose=3)
gbm2.fit(X_train, y_train.f_31)

# Fill NaN values in f_31 using predictions
train_data.loc[train_data['f_31'].isna(), 'f_31'] = gbm2.predict(X_train_na)
valid_data.loc[valid_data['f_31'].isna(), 'f_31'] = gbm2.predict(X_valid_na)
test_data.loc[test_data['f_31'].isna(), 'f_31'] = gbm2.predict(X_test_na)

cnt_na = np.sum(train_data.isna())
cols = cnt_na[cnt_na != 0].index
fillna_dict = {c: np.mean(train_data[c]) for c in cols if c not in ['f_30', 'f_31']}

# Fill remaining NaN values using fillna_dict for other columns
train_data.fillna(fillna_dict, inplace=True)
valid_data.fillna(fillna_dict, inplace=True)
test_data.fillna(fillna_dict, inplace=True)

# Scale features
mms = MinMaxScaler(feature_range=(0, 1))
dense_feature = ['f_' + str(i) for i in range(42, 80)]
train_data[dense_feature] = mms.fit_transform(train_data[dense_feature])
valid_data[dense_feature] = mms.transform(valid_data[dense_feature])
test_data[dense_feature] = mms.transform(test_data[dense_feature])

# Combine training and validation data for final DataFrame
train_data_full = pd.concat([train_data, valid_data], axis=0).reset_index(drop=True)


# Define the dataset directory
dataset_name = '/kaggle/working/recs_challenge_2023/data/final'

# Create the directory if it doesn't exist
os.makedirs(dataset_name, exist_ok=True)

# Save data
train_data.to_csv(f'{dataset_name}/train_data.csv', index=None)
test_data.to_csv(f'{dataset_name}/test_data.csv', index=None)
valid_data.to_csv(f'{dataset_name}/valid_data.csv', index=None)
train_data_full.to_csv(f'{dataset_name}/train_data_full.csv', index=None)