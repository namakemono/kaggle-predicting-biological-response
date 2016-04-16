import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

"""
# References
- http://yag.xyz/blog/2015/08/08/xgboost-python/
"""

def build(df):
    X, y = [], []
    for idx in df.index:
        row = df.ix[idx]
        x = []
        for column in df.columns:
            if "Activity" in column:
                y.append(float(row[column]))
            else:
                x.append(float(row[column]))
        X.append(x)
    return X, y

def f(params):
    score_list = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test) 
        watchlist = [(dtrain, 'train'),(dvalid, 'eval')]  
        model = xgb.train(params, dtrain, num_boost_round=1500, evals=watchlist, early_stopping_rounds=50, verbose_eval=False)
        y_preda = model.predict(dvalid)
        y_pred = [1 if _ > 0.5 else 0 for _ in y_preda]
        score = accuracy_score(y_test, y_pred)
        score_list.append(score)
    loss = -np.array(score_list).mean()
    print "AVG ACC: %.2f%%" % (-100 * loss)
    print params
    return {'loss': loss, 'status': STATUS_OK}

def optimize(trials):
    space = {
        "objective": "binary:logistic",
        "eval_metric": "logloss", # "error",
        "eta" : hp.quniform("eta", 0.2, 0.6, 0.05),
        "max_depth" : hp.quniform("max_depth", 1, 10, 1),
        "min_child_weight" : hp.quniform('min_child_weight', 1, 10, 1),
        'gamma' : hp.quniform('gamma', 0, 1, 0.05),
        "subsample" : hp.quniform('subsample', 0.5, 1, 0.05),
        "colsample_bytree" : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'silent' : 1,
    }
    best = fmin(f, space, algo=tpe.suggest, trials=trials, max_evals=50)
    print "best parameters",best
    print f(best)

def run(is_validation=True):
    train_df = pd.read_csv("../inputs/train.csv")
    test_df = pd.read_csv("../inputs/test.csv")
    X_all, y_all = build(train_df)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        'eta': 0.2,
        'max_depth': 10,
        'min_child_weight': 9.0,
        'gamma': 0.75,
        'subsample': 0.9,
        'colsample_bytree': 0.5,
        'silent': 1,
    }
    if is_validation:
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=1) 
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
        model = xgb.train(params, dtrain, num_boost_round=1500, evals=watchlist, early_stopping_rounds=50, verbose_eval=False)
        y_preda = model.predict(dvalid)
        y_pred = [1 if p > 0.5 else 0 for p in y_preda]
        print "Acc: %f, Cross Entropy Loss: %f" % (accuracy_score(y_test, y_pred), log_loss(y_test, y_preda))
    else:
        X_train, y_train = X_all, y_all
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain, num_boost_round=1500, verbose_eval=False)
        X_test, y_test = build(test_df)
        dvalid = xgb.DMatrix(X_test, label=y_test)
        y_preda = model.predict(dvalid)
        pd.DataFrame([{"MoleculeId": (i+1), "PredictedProbability": p} for i, p in enumerate(y_preda)]).to_csv("../outputs/submissions.csv", index=False)    

if __name__ == "__main__":
    run(True)
    """
    train_df = pd.read_csv("../inputs/train.csv")
    X, y = build(train_df)
    trials = Trials()
    optimize(trials)
    """

