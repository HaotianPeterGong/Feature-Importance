import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import clone
import shap


def spearman_rank_corr(df: pd.DataFrame, target:str) -> dict:
    """Return the spearman's rank correlations for each feature column 
    given target column in pd.df"""
    df_copy = df.copy()
    rank_target = df_copy[target].rank(axis=0, method='min') # get min ranking num for same value
    std_target = rank_target.std()
    cols = [col for col in df.columns if col != target]
    corr = {}
    for col in cols:
        rank_col = df_copy[col].rank()
        cov = rank_col.cov(rank_target)
        std_col = rank_col.std()
        corr_col = abs(cov / (std_col*std_target))
        corr[col] = corr_col
    return dict(sorted(corr.items(), key=lambda item: item[1], reverse=True))


def cleveland_dot_plot(data, title, error=None, datalabels=None, ax=None, 
             marker='.', markersize=12, **kwargs):
    if ax is None:
        ax = plt.gca()

    try:
        n = len(data)
    except ValueError:
        n = data.size

    y = np.arange(n)[::-1]

    l = ax.plot(data, y, marker=marker, color='#08D4DC', 
                linestyle='', markersize=markersize, 
                markeredgewidth=0, **kwargs)

    if error is not None:
        lo = data - error
        hi = data + error

        l1 = ax.hlines(y, lo, hi, color=l[0].get_color())
        l.append(l1)

    y_ticks = ax.yaxis.set_ticks(range(n))
    y_text = ax.yaxis.set_ticklabels(datalabels[::-1], fontsize=12)

    ax.set_ylim(-1, n)

    ax.tick_params(axis='y', which='major', left='on', color='0.8')
    ax.grid(axis='y', which='major', color='0.8', zorder=-10, linestyle='--')
    
    ax.set_title(title, y=1.02, color ='#2F2F3B', fontsize=14, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return l


def draw_cleveland_plot(corr_dict: dict, plot_title: str, 
                        figsize=(10, 8)):
    """Draw cleveland dot plot given correlation dict"""
    x = [v for v in corr_dict.values()]
    categories = [key for key in corr_dict.keys()]
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    l = cleveland_dot_plot(x, plot_title, 
                           datalabels=categories, markersize=12)


def mRMR_corr(df: pd.DataFrame, target:str) -> dict:
    """return MRMR correlation in a dictionary given df and target column name"""
    df_copy = df.copy()
    corr_all_dict = spearman_rank_corr(df_copy, target)
    feature_cols = [key for key in corr_all_dict.keys()]
    mrmr_dict = {}
    for col in feature_cols:
        relevance = corr_all_dict[col]
        redundancy_df = df_copy.iloc[:, :-1]\
                                     .corr(method='spearman')[col].reset_index()
        redundancy_df = redundancy_df[(redundancy_df['index'] != col) 
                                      & (redundancy_df['index'] != target)]\
                                     .reset_index(drop=True)
        redundancy = redundancy_df[col].mean()
        mrmr = relevance - redundancy
        mrmr_dict[col] = mrmr
    return dict(sorted(mrmr_dict.items(), key=lambda item: item[1], reverse=True))


def dropcol_importances(model, X_train: pd.DataFrame, 
                        y_train, X_val, y_val,
                        metric) -> dict:
    """
    1. Compute validation metric for model trained on all features 
    2. Drop column xj from training set 
    3. Retrain model 
    4. Compute validation metric set 
    5. Importance score is the change in metric
    """
    model.fit(X_train, y_train)
    baseline = metric(y_val, model.predict(X_val))
    imp = {}
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_val_ = X_val.drop(col, axis=1)
        model_ = clone(model)
        model_.fit(X_train_, y_train)
        m = metric(y_val, model_.predict(X_val_))
        imp[col] = baseline - m
    return dict(sorted(imp.items(), key=lambda item: item[1], reverse=True))


def permutation_importances(model, X_train: pd.DataFrame, 
                            y_train, X_val, y_val,
                            metric) -> dict:
    """
    1. Compute validation metric for model trained on all features 
    2. Permute column xj in validation set 
    3. Compute validation metric set 
    4. Importance score is the change in metric
    """
    model.fit(X_train, y_train)
    baseline = metric(y_val, model.predict(X_val))
    imp = {}
    for col in X_val.columns:
        col_save = X_val[col].copy()
        X_val[col] = np.random.permutation(X_val[col])
        m = metric(y_val, model.predict(X_val))
        X_val[col] = col_save
        imp[col] = baseline - m
    return dict(sorted(imp.items(), key=lambda item: item[1], reverse=True))


def cumulate_feature_loss(model, X_train: pd.DataFrame,
                          y_train, X_val, y_val, feat_imp: dict, 
                          top_feature=10, 
                          loss_metric=log_loss) -> list:
    """
    Get the top n features in feat_imp, and return the log loss
    for range(1, n+1) top features
    """
    loss_list = []
    for i in range(1, top_feature+1):
        model_ = clone(model)
        features = [key for key in feat_imp][:i]
        model_.fit(X_train[features], y_train)
        logit = model_.predict_proba(X_val[features])
        loss = loss_metric(y_val, logit)
        loss_list.append(loss)
    return loss_list


def feat_select(model, X_train: pd.DataFrame,
                y_train, X_val, y_val, feat_imp: dict,
                loss_metric=log_loss) -> list:
    """
    We need a mechanism to drop off unimportant features and keep the top k, 
    for some k we don't know beforehand. 
    Implement an automated mechanism that selects the top k features automatically 
    that gives the best validation error. 
    In other words, get a baseline validation metric appropriate for a classifier 
    or a regressor then get the feature importances. 
    Dropped the lowest importance feature and retrain the model and re-computing the validation metric. 
    If the validation metric is worse, then we have dropped one too many features. 
    Because of codependencies between features, 
    you must recompute the feature importances after dropping each feature.
    """
    model_ = clone(model)
    model_.fit(X_train, y_train)
    base_logit = model_.predict_proba(X_val)
    base_loss = loss_metric(y_val, base_logit)

    feat_del_list = []
    loss_last = base_loss
    for i in range(len(X_train.columns)):
        col_list = [key for key in feat_imp]
        feat_del = col_list[-1] # already sorted
        feat_del_list.append(feat_del)
        feat_keep_list = col_list[:-1]
        print(f"Loop {i+1}, drop feature: {feat_del}")

        model_ = clone(model)
        X_train_new = X_train[feat_keep_list]
        X_val_new = X_val[feat_keep_list]
        model_.fit(X_train_new, y_train)
        logit = model_.predict_proba(X_val_new)
        loss = loss_metric(y_val, logit)
        if loss <= loss_last:
            print(f"\tLoss decrease: {loss_last:.4} -> {loss:.4}")
            loss_last = loss
            shap_explainer = shap.TreeExplainer(model_, data=X_train_new)
            shap_value = shap_explainer.shap_values(X=X_val_new, 
                                                    y=y_val, check_additivity=False)
            shap_imp_value = np.sum(np.mean(np.abs(shap_value), axis=1), axis=0)
            feat_imp = dict(zip(X_train_new.columns, shap_imp_value))
            feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
            # replace feat_imp with shap_imp at the first loop
        else:
            print(f"\tLoss increase: {loss_last:.4} -> {loss:.4}")
            break
    print(f"\nDrop features: {feat_del_list[:-1]}")
    print(f"Select features: {feat_keep_list+[feat_del]}")
    
    return feat_keep_list


def feat_std(model, X_train: pd.DataFrame,
             y_train, X_val, y_val, boostrap_num=100) -> dict:
    """
    Return sorted dict of shap feature importance std,
    boostrapping y_train 100 times
    """
    model.fit(X_train, y_train)
    shap_explainer = shap.TreeExplainer(model, data=X_train)
    shap_imp_array = np.zeros((100, X_val.shape[1]))
    for i in range(boostrap_num):
        idx = np.random.choice(range(y_val.shape[0]), 
                               size=y_val.shape[0], replace=True)
        boostrap_y_val = y_val.iloc[idx]

        shap_value = shap_explainer.shap_values(X=X_val, y=boostrap_y_val, 
                                                check_additivity=False)
        shap_imp_value = np.sum(np.mean(np.abs(shap_value), axis=1), axis=0)
        shap_imp_array[i] = shap_imp_value
    shap_imp_std = np.std(shap_imp_array, axis=0)
    shap_imp_std_dict = dict(zip(X_train.columns, shap_imp_std))
    shap_imp_std_dict = dict(sorted(shap_imp_std_dict.items(), 
                                    key=lambda item: item[1], reverse=True))
    return shap_imp_std_dict


def get_p_values(model, X_train: pd.DataFrame,
             y_train, X_val, y_val, 
             simulation_num=500) -> (np.array, np.array, np.array):
    """
    shuffle the target variable y, and then compute the 
    feature importances again. Count how many times a feature 
    is as important or more important than the true 
    feature importance computed as a baseline.
    """
    model.fit(X_train, y_train)
    shap_explainer = shap.TreeExplainer(model, data=X_train)
    base_shap_values = shap_explainer.shap_values(X=X_val, y=y_val, 
                                                  check_additivity=False)
    base_shap_values = np.sum(np.mean(np.abs(base_shap_values), axis=1), axis=0)
    shap_baseline = base_shap_values / np.sum(base_shap_values) # normalize
    
    shap_imp_array = np.zeros((simulation_num, X_val.shape[1]))
    for i in range(simulation_num):
        model_ = clone(model)
        y_train_ = np.random.permutation(y_train)
        model_.fit(X_train, y_train_)
        shap_explainer = shap.TreeExplainer(model_, data=X_train)
        shap_values = shap_explainer.shap_values(X=X_val, y=y_val, 
                                                  check_additivity=False)
        shap_imp_array[i] = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
        shap_imp_array[i] = shap_imp_array[i] / np.sum(shap_imp_array[i])
    diff = shap_imp_array - shap_baseline
    p_values = np.sum(diff >= 0, axis=0) / simulation_num
    return p_values, shap_baseline, shap_imp_array


def get_significant_col(p_values: np.array, 
                        X_train: pd.DataFrame, thred=0.05) -> dict:
    "return dictionary of {(significant col, p_value) :order}"
    p_value_list = list(zip(X_train.columns, p_values))
    order = [i for i in range(len(p_value_list)) 
             if p_value_list[i][1] < thred]
    sig_pair = [(f, p) for f, p in p_value_list if p < thred]
    return dict(zip(sig_pair, order))
