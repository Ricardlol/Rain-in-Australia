from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_curve, average_precision_score, \
    roc_auc_score, roc_curve, auc, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import time
from sklearn.preprocessing import PolynomialFeatures
pd.set_option("display.max_columns", None)


def analyseData(database):
    print("Informacio de la base de dades:")
    print(database.info())
    print("---------\ncapçalera:")
    print(database.head())
    sns.heatmap(database.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()
    print("---------\npercentatges de nuls en cada variable:")
    missing = pd.DataFrame(database.isnull().sum(), columns=['No. of missing values'])
    missing['% missing_values'] = (missing / len(database)).round(2) * 100
    print(missing)

    continuous_cols = list(database.select_dtypes(include=['float64']).columns)
    plt.figure(figsize=(12, 10))
    database.boxplot(continuous_cols, rot=90)

    print("---------\nclass in target:")
    print(database['RainTomorrow'].value_counts())
    # Fer proves amb la columna resultat

# option pot escollir Yes per transormar els nulls a la clase yes,
# si es posa No pasara els valos nulls a la clase no, en cualsevol altre cas els eliminninara
def fixTarget(df, option = "No"):
    if (option == 'No'):
        df['RainTomorrow'] = [1 if i == 'Yes' else 0 for i in df['RainTomorrow']]
    elif (option == 'Yes'):
        df['RainTomorrow'] = [0 if i == 'No' else 1 for i in df['RainTomorrow']]
    else:
        df = df.dropna(axis=0, subset=['RainTomorrow'])
    print("Opcio escollida:", option)
    print(df['RainTomorrow'].value_counts())
    print("--------------------------------------")
    return df

def fixMissingValuesMode(df):
    variables = list(df.select_dtypes(include=['float64', 'object']).columns)
    xModedf = df.copy()
    for i in variables:
        xModedf[i].fillna(xModedf[i].mode()[0], inplace=True)
    return xModedf

def fixMissingValuesMedian(df):
    variables = list(df.select_dtypes(include=['float64', 'object']).columns)
    xMediandf = df.copy()
    for i in variables:
        if (np.dtype(xMediandf[i]) == 'object'):
            xMediandf[i].fillna(xMediandf[i].mode()[0], inplace=True)
        else:
            xMediandf[i].fillna(xMediandf[i].median(), inplace=True)
    return xMediandf

def fixMissingValuesMean(df):
    variables = list(df.select_dtypes(include=['float64', 'object']).columns)
    xMeandf = df.copy()
    for i in variables:
        if (np.dtype(xMeandf[i]) == 'object'):
            xMeandf[i].fillna(xMeandf[i].mode()[0], inplace=True)
        else:
            xMeandf[i].fillna(xMeandf[i].mean(), inplace=True)
    return xMeandf

def oversamplingRandom(df):
    count_class_0, count_class_1 = df.RainTomorrow.value_counts()
    df_class_0 = df[df['RainTomorrow'] == 0]
    df_class_1 = df[df['RainTomorrow'] == 1]

    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df = pd.concat([df_class_0, df_class_1_over], axis=0)
    return df

def undersamplingRandom(df):
    count_class_0, count_class_1 = df.RainTomorrow.value_counts()

    df_class_0 = df[df['RainTomorrow'] == 0]
    df_class_1 = df[df['RainTomorrow'] == 1]

    df_class_0_under = df_class_0.sample(count_class_1)
    df = pd.concat([df_class_0_under, df_class_1], axis=0)
    return df

def oversamplingSMOTE(df, x, y):
    smote = SMOTE()
    x, y = smote.fit_resample(x, y)

    print(x.shape, y.shape)
    return x, y

def EnchanceData(x):
    # esborrant dia (dada identificadora, no vàlides pels models)
    variablesCategoric = list(x.select_dtypes(include=['object']).columns)
    # transformer = ColumnTransformer(
    #                         transformers=[('notCategoric',
    #                                         OrdinalEncoder(sparse='False', drop='first'),
    #                                        variablesCategoric)],
    #                         remainder='passthrough')
    transformer = ColumnTransformer(
        transformers=[('notCategoric',
                       OneHotEncoder(sparse='False', drop='first'),
                       variablesCategoric)],
        remainder='passthrough')
    return transformer.fit_transform(x)


def standarise(df, with_mean=False):
    # scaler = preprocessing.MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    return df

def balanceData(X, y):
    # https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    # bàsicament pel fet que un 80% de les Y són 0 i un 20% són 1
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    return X, y

def removeOutliers(df):
    return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

def check_skewness(df):
    skew_limit = 0.75
    skew_value = df[df.columns].skew()
    skew_cols = skew_value[abs(skew_value) > skew_limit]
    cols = skew_cols.index
    return cols

def transformutilsColumns(X,liersSkew):
    pt = PowerTransformer(standardize=False)
    X[liersSkew] = pt.fit_transform(X[liersSkew])
    return X


def logisticRegression(X_test, X_train, y_test, y_train, proba=False):
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    logireg.fit(X_train, y_train.values.ravel())  # https://www.geeksforgeeks.org/python-pandas-series-ravel/
    if proba:
        y_pred = logireg.predict_proba(X_test)
        # return y_pred

    y_pred = logireg.predict(X_test)
    print("\nLogistic")
    printMetrics(y_pred, y_test)


def svcLinear(X_test, X_train, y_test, y_train):
    # https://scikit-learn.org/stable/modules/svm.html#complexity
    svc = svm.LinearSVC(C=2,max_iter=500, penalty="l2",random_state=0, tol=1e-4)
    svc.fit(X_train, y_train.values.ravel())
    y_pred = svc.predict(X_test)
    print("\nSVC Linear")
    printMetrics(y_pred, y_test)

def svc(X_test, X_train, y_test, y_train, proba=False, kernels=['rbf']):
    for kernel in kernels:
        svc = svm.SVC(C=100, kernel=kernel, probability=True, random_state=0)
        svc.fit(X_train.head(1000), y_train.head(1000).values.ravel())
        if proba:
            y_pred = svc.predict_proba(X_test.head(100))
            return y_pred
        y_pred = svc.predict(X_test.head(100))
        print("\nSVC")
        printMetrics(y_pred, y_test.head(100))


def xgbc(X_test, X_train, y_test, y_train, proba=False):
    xgbc = XGBClassifier(objective='binary:logistic', use_label_encoder =False, n_estimators=5,gamma=0.5, random_state=0)
    xgbc.fit(X_train,y_train.values.ravel())
    if proba:
        y_pred = xgbc.predict_proba(X_test)
        return y_pred

    y_pred = xgbc.predict(X_test)
    print("\nXGBC")
    printMetrics(y_pred, y_test)



def rfc(X_test, X_train, y_test, y_train, proba=False):
    clf = RandomForestClassifier(max_leaf_nodes=15,n_estimators=100, ccp_alpha=0.0,bootstrap=True, random_state=0)
    clf.fit(X_train,y_train)
    if proba:
        y_pred = clf.predict_proba(X_test)
        return y_pred

    y_pred = clf.predict(X_test)
    print("\nRandom Forest")
    printMetrics(y_pred, y_test)


def knn(X_test, X_train, y_test, y_train, neighbors=2, proba=False):
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights="uniform", p=2)
    knn.fit(X_train, y_train)
    if proba:
        y_pred = knn.predict_proba(X_test)
        return y_pred

    y_pred = knn.predict(X_test)
    # print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))


def decicionTree(X_test, X_train, y_test, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("\nDecision Tree")
    printMetrics(y_pred, y_test)


def baggingDecicionTree(X_test, X_train, y_test, y_train):
    bagDt = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                                bootstrap=True, n_jobs=-1, oob_score=True)
    bagDt.fit(X_train, y_train)
    y_pred = bagDt.predict(X_test)
    print("\nBagging (Decision Tree)")
    printMetrics(y_pred, y_test)


def printMetrics(y_pred, y_test):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))


def baggingRandomForest(X_test, X_train, y_test, y_train):
    bagDt = BaggingClassifier(RandomForestClassifier(), n_estimators=40, max_samples=1250,
                                bootstrap=True, n_jobs=-1, oob_score=True)
    bagDt.fit(X_train, y_train)
    y_pred = bagDt.predict(X_test)
    print("\nBagging (Random Forest)")
    printMetrics(y_pred, y_test)


def baggingXGBC(X_test, X_train, y_test, y_train):
    bagDt = BaggingClassifier(XGBClassifier(objective='binary:logistic', use_label_encoder =False, n_estimators=5,gamma=0.5, random_state=0), n_estimators=40, max_samples=1250,
                                bootstrap=True, n_jobs=-1, oob_score=True)
    bagDt.fit(X_train, y_train)
    y_pred = bagDt.predict(X_test)
    print("\nBagging (XGBClassifier)")
    printMetrics(y_pred, y_test)


def kfold(X, y):
    for k in range(2, 7):
        kf = KFold(n_splits=k)
        print(kf.get_n_splits(X))
        for train_index, test_index in kf.split(X):
         X_train, X_test = X[train_index], X[test_index]
         y_train, y_test = y[train_index], y[test_index]
         X_train, y_train = balanceData(X_train, y_train)
         # logisticRegression(X_test, X_train, y_test, y_train)
         # svc(X_test, X_train, y_test, y_train, False, ["poly"])
         # rfc(X_test, X_train, y_test, y_train)
         # xgbc(X_test, X_train, y_test, y_train)
         # svcLinear(X_test, X_train, y_test, y_train)
         knn(X_test, X_train, y_test, y_train, 2)

def strack_modle(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = balanceData(X_train, y_train)
    base_models = [('xgb',
                    XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=5, gamma=0.5,
                                  random_state=0)), ('lr', LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001))]
    meta_model = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=True, cv=5)

    stacking_model.fit(X_train, y_train)
    y_pred= stacking_model.predict(X_test)
    print(f1_score(y_test, y_pred))

def main():
    database = pd.read_csv('./data/weatherAUS.csv')

    analyseData(database)

    database = fixTarget(database, "No")

    database = oversamplingRandom(database)
    # database = undersamplingRandom(database)

    x = database.drop(['RainTomorrow'], axis=1)
    y = database['RainTomorrow']

    cols_to_drop = ['Date']
    database.drop(columns=cols_to_drop, inplace=True)

    x = fixMissingValuesMode(x)
    # x = fixMissingValuesMedian(x)
    # x = fixMissingValuesMean(x)

    x = EnchanceData(x)

    # database = oversamplingSMOTE(database)
    x = pd.DataFrame(x.toarray())
    skewed_col = check_skewness(x)

    x = transformutilsColumns(x, skewed_col)


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train = standarise(X_train)
    X_test = standarise(X_test)

    inicio = time.time()
    logisticRegression(X_test, X_train, y_test, y_train)
    tempsTrigat = time.time() - inicio;
    print(tempsTrigat)

    inicio = time.time()
    svc(X_test, X_train, y_test, y_train, False, ["poly"])
    tempsTrigat = time.time() - inicio;
    print(tempsTrigat)

    inicio = time.time()
    xgbc(X_test, X_train, y_test, y_train)
    tempsTrigat = time.time() - inicio;
    print(tempsTrigat)

    inicio = time.time()
    baggingXGBC(X_test, X_train, y_test, y_train)
    tempsTrigat = time.time() - inicio;
    print(tempsTrigat)

    inicio = time.time()
    rfc(X_test, X_train, y_test, y_train)
    tempsTrigat = time.time() - inicio;
    print(tempsTrigat)

    inicio = time.time()
    baggingRandomForest(X_test, X_train, y_test, y_train)
    tempsTrigat = time.time() - inicio;
    print(tempsTrigat)

    inicio = time.time()
    svcLinear(X_test, X_train, y_test, y_train)
    tempsTrigat = time.time() - inicio;
    print(tempsTrigat)

    inicio = time.time()
    decicionTree(X_test, X_train, y_test, y_train)
    tempsTrigat = time.time() - inicio;
    print(tempsTrigat)

    inicio = time.time()
    baggingDecicionTree(X_test, X_train, y_test, y_train)
    tempsTrigat = time.time() - inicio;
    print(tempsTrigat)

    scores = cross_val_score(LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001), x, y, cv=5, scoring="f1_macro")
    # scores = cross_val_score(svm.SVC(C=2, kernel="poly", probability=False, random_state=0, tol=0.0001, max_iter=500), X, y, cv=5,scoring="f1_macro")
    # scores = cross_val_score(RandomForestClassifier(max_leaf_nodes=15,n_estimators=100, ccp_alpha=0.0,bootstrap=True, random_state=0),X, y, cv=5, scoring="f1_macro")
    # scores = cross_val_score(XGBClassifier(objective='binary:logistic', use_label_encoder =False, n_estimators=5,gamma=0.5, random_state=0),X, y, cv=5, scoring="f1_macro")
    # scores = cross_val_score(
    #     svm.LinearSVC(C=2,max_iter=500, penalty="l2",random_state=0, tol=1e-4),
    #     X, y, cv=5, scoring="f1_macro")
    # scores = cross_val_score(
    #     KNeighborsClassifier(n_neighbors=2, weights="uniform", p=2),
    #     X, y, cv=5, scoring="f1_macro")
    print(scores)



#names = ['Date','Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RainTomorrow']
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()