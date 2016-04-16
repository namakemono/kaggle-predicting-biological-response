import glob
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier

def level1_rf():
    for decompose_type in ["raw", "pca", "tfidf"]:
        level1("%s_rf" % decompose_type, RandomForestClassifier, decompose_type=decompose_type, is_calibration=True)

def level1_lr():
    for decompose_type in ["raw", "pca", "tfidf"]:
        level1("%s_lr" % decompose_type, LogisticRegression, decompose_type=decompose_type, is_calibration=True)

def level1_xgboost():
    for decompose_type in ["raw", "pca", "tfidf"]:
        level1("%s_xgb" % decompose_type, XGBClassifier, decompose_type=decompose_type, is_calibration=False)

def level1_svm():
    for decompose_type in ["raw", "pca", "tfidf"]:
        level1("%s_svm" % decompose_type, SVC, decompose_type=decompose_type, is_calibration=False, params={"probability": True})

def level1_et():
    for decompose_type in ["raw", "pca", "tfidf"]:
        level1("%s_et" % decompose_type, ExtraTreesClassifier, decompose_type=decompose_type, is_calibration=True)

def to_filepath(decompose_type, is_train):
    res = "../inputs/"
    if decompose_type != "raw":
        res += decompose_type + "_"
    if is_train:
        res += "train.csv"
    else:
        res += "test.csv"
    return res

def to_filepath(decompose_type, is_train):
    res = "../inputs/"
    if decompose_type != "raw":
        res += decompose_type + "_"
    if is_train:
        res += "train.csv"
    else:
        res += "test.csv"
    return res

def level1(name, ClassifierClass, decompose_type="raw", is_calibration=False, params=None):
    now = datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
    output_train_filepath = "../inputs/level1/%s::%s_train.csv" % (now, name)
    output_test_filepath = "../inputs/level1/%s::%s_test.csv" % (now, name)
    
    train_df = pd.read_csv(to_filepath(decompose_type, is_train=True))
    y_all = train_df["Activity"].values
    del train_df["Activity"]
    X_all = train_df.values
    skf = StratifiedKFold(y_all, n_folds=3)
    y = np.zeros(len(y_all))
    for train_index, test_index in skf:
        X_train, y_train = X_all[train_index], y_all[train_index]
        X_test, y_test = X_all[test_index], y_all[test_index]
        if params is not None:
            clf = ClassifierClass(**params)
        else:
            clf = ClassifierClass()
        if is_calibration:
            clf = CalibratedClassifierCV(clf, method='isotonic', cv=10)
        clf.fit(X_train, y_train)
        y_preda = clf.predict_proba(X_test)[:,1]
        print "[Lv.1][%s]Acc: %.3f, Log Loss: %.3f" % (name, clf.score(X_test, y_test), log_loss(y_test, y_preda))
        y[test_index] = y_preda
    pd.DataFrame([{"MoleculeId": (i+1), "%s-PredictedProbability" % name: p} for i, p in enumerate(y)]).to_csv(output_train_filepath, index=False)

    if params is not None:
        clf = ClassifierClass(**params)
    else:
        clf = ClassifierClass()
    if is_calibration:
        clf = CalibratedClassifierCV(clf, method='isotonic', cv=10)
    clf.fit(X_all, y_all)
    test_df = pd.read_csv(to_filepath(decompose_type, is_train=False))
    X_test = test_df.values
    y_preda = clf.predict_proba(X_test)[:,1]
    pd.DataFrame([{"MoleculeId": (i+1), "PredictedProbability": p} for i, p in enumerate(y_preda)]).to_csv(output_test_filepath, index=False)

# Level 2
def level2_xgboost():
    ClassifierClass = XGBClassifier
    train_df = pd.read_csv("../inputs/train.csv")["Activity"]
    for filepath in glob.glob("../inputs/level1/*_train.csv"):
        df = pd.read_csv(filepath) 
        del df["MoleculeId"]
        train_df = pd.concat([train_df, df], axis=1)
    y_all = train_df["Activity"].values
    del train_df["Activity"] 
    X_all = train_df.values
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=1)
    clf = ClassifierClass()
    clf.fit(X_train, y_train)
    y_preda = clf.predict_proba(X_test)[:,1]
    print "[Lv.2][xgb]Acc: %.3f, Log Loss: %.3f" % (clf.score(X_test, y_test), log_loss(y_test, y_preda))
    
    clf = ClassifierClass()
    clf.fit(X_all, y_all)
    test_df = None
    for filepath in glob.glob("../inputs/level1/*_test.csv"):
        df = pd.read_csv(filepath) 
        del df["MoleculeId"]
        test_df = pd.concat([test_df, df], axis=1)
    X_test = test_df.values 
    y_preda = clf.predict_proba(X_test)[:,1]
    now = datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
    pd.DataFrame([{"MoleculeId": (i+1), "PredictedProbability": p} for i, p in enumerate(y_preda)]).to_csv("../outputs/%s::submissions.csv" % now, index=False)
 
def run():
    # level1_svm()
    """
    level1_xgboost() # 0.43210   
    level1_lr() # 0.53313
    level1_rf() # 0.85940
    level1_et() # 0.42726
    """
    level2_xgboost() # 0.48221

if __name__ == "__main__":
    run()
