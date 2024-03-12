import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

from catboost import Pool

def select_duplicates(train, test):
    tmp = pd.concat([train, test])
    dup_cols = tmp.columns[tmp.nunique()==1]
    return dup_cols


def select_corr_columns(train, test, thresh):
    corr_matrix = train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    train_corr_col = [column for column in upper.columns if any(upper[column] > thresh)]

    corr_matrix = test.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    test_corr_col = [column for column in upper.columns if any(upper[column] > thresh)]
    return np.intersect1d(train_corr_col, test_corr_col)


def grouped_train_test_split(X, y, groups, train_size):
    c = Counter(list(groups))
    unique_groups = [key for key, value in c.items()]
    group_counts = [value for key, value in c.items()]
    train_groups, test_groups = train_test_split(unique_groups, train_size=train_size, stratify=group_counts, shuffle=True, random_state=42)
    train_idx = np.argwhere(np.isin(groups, train_groups)).ravel()
    test_idx = np.argwhere(np.isin(groups, test_groups)).ravel()
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], groups.iloc[train_idx], groups.iloc[test_idx]


def grouped_cv_split(groups, n_splits):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    c = Counter(groups)
    unique_groups = np.array([key for key, value in c.items()])
    group_counts = np.array([value for key, value in c.items()])
    cv = list(skf.split(X = unique_groups, y=group_counts))
    for i, split in enumerate(cv):
        split = list(split)
        train_idx = np.argwhere(np.isin(groups, unique_groups[split[0]])).ravel()
        test_idx = np.argwhere(np.isin(groups, unique_groups[split[1]])).ravel()
        split[0] = train_idx
        split[1] = test_idx
        cv[i] = tuple(split)
    return cv


def get_best_thresh(y_val, y_pred):
    dx = (y_pred.max() - y_pred.min()) * 0.001
    thresholds = np.arange(y_pred.min(), y_pred.max() + dx, dx)
    best_threshold = np.nan
    best_auc = -1
    for threshold in thresholds:
        auc_score = roc_auc_score(y_val, (y_pred > threshold).astype(int))
        if auc_score > best_auc:
            best_auc = auc_score
            best_threshold = threshold
    return best_threshold


def ranker_cv_auc_score(model,
                        X_train, y_train, train_id,
                        X_val, y_val, val_id,
                        cat_columns):
    scores = np.zeros(5, dtype=float)
    for i, (train_index, test_index) in enumerate(grouped_cv_split(train_id, 5)):
        train = Pool(
            data=X_train.iloc[train_index],
            cat_features=list(cat_columns.values),
            label=y_train.iloc[train_index],
            feature_names=list(X_train.columns.values),
            group_id=train_id.iloc[train_index]
        )
        val = Pool(
            data=X_val,
            cat_features=list(cat_columns.values),
            label=y_val,
            feature_names=list(X_val.columns.values),
            group_id=val_id
        )
        test = Pool(
            data=X_train.iloc[test_index],
            cat_features=list(cat_columns.values),
            label=y_train.iloc[test_index],
            feature_names=list(X_train.columns.values),
            group_id=train_id.iloc[test_index]
        )

        model.fit(train, eval_set=val)
        
        thresh = get_best_thresh(
            y_val.values,
            model.predict(val)
        )
        scores[i] = roc_auc_score(
            y_train.iloc[test_index].values,
            (model.predict(test) > thresh).astype(int)
        )
    return scores


def ranker_cv_ndcg_score(model,
                         X_train, y_train, train_id,
                         X_val, y_val, val_id,
                         cat_columns):
    scores = np.zeros(5, dtype=float)
    for i, (train_index, test_index) in enumerate(grouped_cv_split(train_id, 5)):
        train = Pool(
            data=X_train.iloc[train_index],
            cat_features=list(cat_columns.values),
            label=y_train.iloc[train_index],
            feature_names=list(X_train.columns.values),
            group_id=train_id.iloc[train_index]
        )
        val = Pool(
            data=X_val,
            cat_features=list(cat_columns.values),
            label=y_val,
            feature_names=list(X_val.columns.values),
            group_id=val_id
        )
        test = Pool(
            data=X_train.iloc[test_index],
            cat_features=list(cat_columns.values),
            label=y_train.iloc[test_index],
            feature_names=list(X_train.columns.values),
            group_id=train_id.iloc[test_index]
        )

        model.fit(train, eval_set=val)
        
        scores[i] = model.score(test)
    return scores