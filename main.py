import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostRanker, Pool
from utils import (
    select_duplicates,
    select_corr_columns,
    grouped_train_test_split
)


train_df = pd.read_csv('train_df.csv')
X_train = train_df.drop(columns=['search_id', 'target'])
train_id = train_df['search_id']
y_train = train_df['target']

test_final_df = pd.read_csv('test_df.csv')
X_test_final = test_final_df.drop(columns=['search_id', 'target'])
test_final_id = test_final_df['search_id']
y_test_final = test_final_df['target']


dup_cols = select_duplicates(X_train, X_test_final)
X_train.drop(columns=dup_cols, inplace=True)
X_test_final.drop(columns=dup_cols, inplace=True)


corr_cols = select_corr_columns(X_train, X_test_final, 0.9)
X_train.drop(columns=corr_cols, inplace=True)
X_test_final.drop(columns=corr_cols, inplace=True)


cat_columns = X_train.columns[X_train.dtypes==np.int64]
num_columns = X_train.columns[X_train.dtypes==np.float64]


oe = OrdinalEncoder(handle_unknown='use_encoded_value',
                    unknown_value=np.nan)
X_train.iloc[:] = oe.fit_transform(X_train)
X_test_final[:] = oe.transform(X_test_final)
X_train[cat_columns] = X_train[cat_columns].astype(np.int64)
X_test_final[cat_columns] = X_test_final[cat_columns].astype(np.int64)


X_train, X_test, y_train, y_test, train_id, test_id = grouped_train_test_split(X_train, y_train, train_id, 0.7)
X_test, X_val, y_test, y_val, test_id, val_id = grouped_train_test_split(X_test, y_test, test_id, 1/3)



params = {
    'early_stopping_rounds': 100,
    'loss_function': 'PairLogitPairwise', 
    'custom_metric': 'NDCG',
    'task_type': 'GPU',
    'bootstrap_type': 'Bernoulli',
    'verbose': False
}
suggested_params = {
    "n_estimators": 50,
    "learning_rate": 0.01670697238865847,
    "depth": 6,
    "subsample": 0.4109271239396233,
    "min_data_in_leaf": 2
}
params.update(suggested_params)
model = CatBoostRanker(**params)


train = Pool(
    data=pd.concat([X_train, X_test]),
    cat_features=list(cat_columns.values),
    label=pd.concat([y_train, y_test]),
    feature_names=list(X_train.columns.values),
    group_id=pd.concat([train_id, test_id])
)
val = Pool(
    data=X_val,
    cat_features=list(cat_columns.values),
    label=y_val,
    feature_names=list(X_val.columns.values),
    group_id=val_id
)
test = Pool(
    data=X_test_final,
    cat_features=list(cat_columns.values),
    label=y_test_final,
    feature_names=list(X_test_final.columns.values),
    group_id=test_final_id
)

model.fit(train, eval_set=val)

print(model.score(test))
