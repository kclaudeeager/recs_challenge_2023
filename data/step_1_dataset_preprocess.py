import numpy as np 
import pandas as pd 
import os 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb


train_data_root='/kaggle/input/train-recsys-challenge-2023'
test_data_file = '/kaggle/input/test-recsys-challenge-2023/000000000000.csv'

train_data_files = os.listdir(train_data_root)
# Load the dataset
# Load the dataset
train_data = [pd.read_csv(os.path.join(train_data_root, train_data_files[i]), sep='\t') 
              for i in range(len(train_data_files))]
train_data = pd.concat(train_data, axis=0)
test_data = pd.read_csv(test_data_file, sep='\t')

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

print("Data Loaded")

print("Initial columns in train_data:", train_data.columns)

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
X_train = train_not_na.drop(['is_clicked', 'is_installed', 'f_30', 'f_31'], axis=1)
y_train = train_not_na[['f_30', 'f_31']]

X_train_na = train_na.drop(['is_clicked', 'is_installed', 'f_30', 'f_31'], axis=1)
X_valid_na = valid_na.drop(['is_clicked', 'is_installed', 'f_30', 'f_31'], axis=1)
X_test_na = test_na.drop(['is_clicked', 'is_installed', 'f_30', 'f_31'], axis=1)

#Train models to fill NaN values
gbm1 = lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42, learning_rate=0.05, max_depth=3, num_leaves=7, verbose=3)
gbm1.fit(X_train, y_train.f_30)
X_train_na['f_30'] = gbm1.predict(X_train_na)
X_valid_na['f_30'] = gbm1.predict(X_valid_na)
X_test_na['f_30'] = gbm1.predict(X_test_na)

gbm2 = lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42, learning_rate=0.05, max_depth=3, num_leaves=7, verbose=3)
gbm2.fit(X_train, y_train.f_31)
X_train_na['f_31'] = gbm2.predict(X_train_na.drop(['f_30'], axis=1))
X_valid_na['f_31'] = gbm2.predict(X_valid_na.drop(['f_30'], axis=1))
X_test_na['f_31'] = gbm2.predict(X_test_na.drop(['f_30'], axis=1))

# Fill NaN values
cnt_na = np.sum(train_data.isna())
cols = cnt_na[cnt_na != 0].index
fillna_dict = {c: np.mean(train_data[c]) for c in cols if c not in ['f_30', 'f_31']}

fill_train = X_train_na[['f_30', 'f_31']]
fill_valid = X_valid_na[['f_30', 'f_31']]
fill_test = X_test_na[['f_30', 'f_31']]

test_data = test_data.fillna(fill_test)
valid_data = valid_data.fillna(fill_valid)
train_data = train_data.fillna(fill_train)

train_data = train_data.fillna(fillna_dict)
valid_data = valid_data.fillna(fillna_dict)
test_data = test_data.fillna(fillna_dict)

# Scale features
mms = MinMaxScaler(feature_range=(0, 1))
dense_feature = ['f_' + str(i) for i in range(42, 80)]
train_data[dense_feature] = mms.fit_transform(train_data[dense_feature])
valid_data[dense_feature] = mms.transform(valid_data[dense_feature])
test_data[dense_feature] = mms.transform(test_data[dense_feature])

# Combine training and validation data for final DataFrame
train_data_full = pd.concat([train_data, valid_data], axis=0).reset_index(drop=True)

# Define the dataset directory
dir_name = '/kaggle/working/recs_challenge_2023/data/final'

# Create the directory if it doesn't exist
os.makedirs(dir_name, exist_ok=True)
dataset_name = 'final'
# Save data
train_data.to_csv(f'{dir_name}/train_data_{dataset_name}.csv', index=None)
test_data.to_csv(f'{dir_name}/test_data_{dataset_name}.csv', index=None)
valid_data.to_csv(f'{dir_name}/valid_data_{dataset_name}.csv', index=None)
train_data_full.to_csv(f'{dir_name}/train_data_{dataset_name}_full.csv', index=None)