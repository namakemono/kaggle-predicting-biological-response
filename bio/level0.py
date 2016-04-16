import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer

def load():
    nrows = None
    train_df = pd.read_csv("../inputs/train.csv", nrows=nrows)
    y_train = train_df["Activity"].values
    del train_df["Activity"]
    X_train = train_df.values
    test_df = pd.read_csv("../inputs/test.csv", nrows=nrows)
    X_test = test_df.values
    return X_train, X_test, y_train

def save(data, columns, filepath):
    print "[SAVE]", filepath
    pd.DataFrame(data, columns=columns).to_csv(filepath, index=False)

def transform():
    name = "tfidf"
    X_train, X_test, y_train = load()
    transformer = TfidfTransformer()
    X_train = transformer.fit_transform(X_train).toarray()
    X_test = transformer.transform(X_test).toarray()
    column_names = ["%s_%d" % (name, (i+1)) for i in range(X_train.shape[1])]
    data = np.c_[y_train, X_train]
    save(np.c_[y_train, X_train], ["Activity"] + column_names, "../inputs/%s_train.csv" % name)
    save(X_test, column_names, "../inputs/%s_test.csv" % name)

def decompose():
    name = "pca"
    n_components = 200
    pca = PCA(n_components=n_components)
    column_names = ["%s_%d" % (name, (i+1)) for i in range(n_components)]
    X_train, X_test, y_train = load()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pca.fit_transform(X_train) 
    X_test = scaler.transform(X_test)
    X_test = pca.transform(X_test)
    save(np.c_[y_train, X_train], ["Activity"] + column_names, "../inputs/%s_train.csv" % name)
    save(X_test, column_names, "../inputs/%s_test.csv" % name)

if __name__ == "__main__":
    transform()
    decompose()
