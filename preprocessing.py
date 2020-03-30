import pandas as pd
from opencage.geocoder import OpenCageGeocode
import geopy.distance
import re
import numpy as np
from collections import defaultdict
from statistics import mean, median
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

def standardizer(Xtrain):
    Xtrain.drop(['daynum','dist_diff','hs_zip','ls_zip'],axis=1,inplace=True)
    Y = Xtrain['Hseed_won']
    cols_to_drop=['HSeed_teamID','LSeed_teamID','Hseed_won','score_diff','year','hs_zip2','ls_zip2']
    X = Xtrain[[i for i in Xtrain.columns if i not in cols_to_drop ]]
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    scaled_X = pd.DataFrame(scaled_X,columns = X.columns)
    return scaled_X

def lin_regression(scaled_X):
    X_const = sm.add_constant(scaled_X)
    est = sm.OLS(Y, X_const)
    estimation_ = est.fit()
    summ = estimation_.summary()

def fwrd_selection(scaled_X,Y):
    # Build RF classifier to use in feature selection
    clf = LogisticRegression()

    sfs1 = sfs(clf,k_features='best',forward=True,
               floating=False,verbose=0,scoring='accuracy',cv=5)
    sfs1 = sfs1.fit(scaled_X,Y)
    feat_cols = list(sfs1.k_feature_idx_)
    fs_vars = [scaled_X.columns[i] for i in feat_cols]
    return fs_vars

xgb_fs = ['seed_diff', 'Adjusted_eff_margin', 'Adjusted_tempo', 'Luck', 'win_pct', 'diff_chmps_L10years',

          'diff_conf_value', 'conf_win_diff', 'last_rank_diff', 'FGA', 'FGA3', 'DR', 'TO', 'PF']
log_fs = ['Adjusted_eff_margin','Adjusted_tempo','Luck','Ajusted_sos',
 'SOS_opp_off','SOS_opp_def','NCSOS','diff_conf_value','last_rank_diff','Stl','PF']
gau_fs = ['Adjusted_eff_margin', 'AdjustedO', 'Luck', 'last_rank_diff']

models = [LogisticRegression(C=1.0),RandomForestClassifier(n_estimators = 100),
          SVC(C=1.0, gamma = 'auto',kernel = 'rbf'),GaussianProcessClassifier(),
          XGBClassifier(n_estimators = 100,gamma = 0.2,subsample=0.9)]

def test_model(models,cols_):
    for model_ in models:
        cv_score = cross_val_score(model_,scaled_X[cols_],Y,cv=5,scoring='accuracy').mean()
    return cv_score