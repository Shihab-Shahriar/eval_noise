import numpy as np
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def create_noise(y,PN,NP):
    yn = y.copy()
    idp = np.argwhere(y==1).ravel()
    idp = np.random.choice(idp,size=int(len(idp)*PN),replace=False)
    yn[idp] = 0
    idp = np.argwhere(y==0).ravel()
    idp = np.random.choice(idp,size=int(len(idp)*NP),replace=False)
    yn[idp] = 1
    return yn

def noisy_evaluate(clf,X,Ytrain,Yeval,CV,eval_metrics):
    scores = defaultdict(list)
    for train_id,test_id in CV.split(X,Ytrain):
        clf.fit(X[train_id],Ytrain[train_id])
        yp = clf.predict(X[test_id])
        for metric in eval_metrics:
            r = metric(Yeval[test_id],yp)
            scores[metric].append(r)
    res = {m:sum(scores[m])/len(scores[m]) for m in eval_metrics}
    return res

def corrupt_label(y,cm):
    a = cm[y]
    s = a.cumsum(axis=1)
    r = np.random.rand(a.shape[0])[:,None]
    yn = (s > r).argmax(axis=1)
    return yn

def correct_noisy_eval(clf,X,Ytrain,Yeval,cm,CV,eval_metrics):
    scores = defaultdict(list)
    Ytrain = corrupt_label(Ytrain,cm)
    for train_id,test_id in CV.split(X,Ytrain):
        clf.fit(X[train_id],Ytrain[train_id])
        yp = clf.predict(X[test_id])
        for metric in eval_metrics:
            r = metric(Yeval[test_id],yp)
            scores[metric].append(r)
    res = {m:sum(scores[m])/len(scores[m]) for m in eval_metrics}
    return res


import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

def read_uci(dataset,stats=False):
    path = f'UCI/{dataset}.txt'
    df = pd.read_csv(path,delim_whitespace=True,header=None)
    df = df.astype('float64')
    data = df.values
    X,Y = data[:,1:],data[:,0].astype('int32')
    if Y.min()==1:
        Y -= 1
    X = MinMaxScaler().fit_transform(X)
    if stats:
        labels,freq = np.unique(Y,return_counts=True)
        print(dataset,X.shape,len(labels),freq.min()/freq.max(),freq)
    return shuffle(X,Y,random_state=42)

from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
def load_mat(name):
    mat = loadmat('../benchmarks.mat')
    X = mat[name][0][0][0]
    Y = mat[name][0][0][1].reshape(-1)
    Y = LabelEncoder().fit_transform(Y)
    X = MinMaxScaler().fit_transform(X)
    return X,Y


def log_loss(wp,X,target,C,PN,NP): 
    """wp=Coefficients+Intercept, X=N*M data matrix, Y=N sized target, C=regularization, PN=p+ or % of Positive samples labeled as Negative
    It is minimized using "L-BFGS-B" method of "scipy.optimize.minimize" function, and results in 
    similar coefficients as sklearn's Logistic Regression when PN=NP=0"""
    c = wp[-1]
    w = wp[:-1]
    z = np.dot(X,w) + c
    yz = target * z    #to compute l(t,y)
    nyz = -target * z  #to compute l(t,-y)
    ls = -log_logistic(yz)   #l(t,y)
    nls = -log_logistic(nyz) #l(t,-y)
    idx = target==1          #indexes of samples w/ P label
    loss = ls.copy()         #To store l-hat
    loss[idx] = (1-NP)*ls[idx] - PN*nls[idx]     #Modified loss for P samples
    loss[~idx] = (1-PN)*ls[~idx] - NP*nls[~idx]  #Modified loss for N samples
    loss = loss/(1-PN-NP) + .5 * (1./C) * np.dot(w, w) #Normalization & regulaqization
    return loss.sum()                             # Final loss

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import _logistic_regression_path, _logistic_loss, _intercept_dot
from sklearn.utils.extmath import log_logistic
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

class Logit(LogisticRegression,BaseEstimator,ClassifierMixin):
    def __init__(self,PN=.2, NP=.2, robust=True,C=np.inf,max_iter=100):
        super().__init__(C=C,max_iter=max_iter)
        self.PN = PN
        self.NP = NP
        self.robust= robust
    
    def fit(self,X,y):
        self.classes_ = unique_labels(y)
        w0 = np.zeros(X.shape[1]+1)
        target = y.copy()
        target[target==0] = -1
        if self.robust:
            self.r_ = minimize(log_loss,w0,method="L-BFGS-B",args=(X, target, self.C,self.PN,self.NP),
                               options={"maxiter": self.max_iter})
        else:
            self.r_ = minimize(_logistic_loss,w0,method="L-BFGS-B",args=(X, target, self.C),options={"maxiter": self.max_iter})
        self.coef_ = self.r_.x[:-1].reshape(1,-1)
        self.intercept_ = self.r_.x[-1:]
        return self

#     def predict(self,X):
#         c = self.r_.x[-1]
#         w = self.r_.x[:-1]
#         z = np.dot(X,w) + c
#         y = z.reshape(-1)
#         y[y<=0] = 0
#         y[y>0] = 1
#         return y
    
from sklearn.datasets import make_classification
def linearly_sep2D(n_samples=800):
    X,y = make_classification(n_samples=200,n_classes=2,n_features=2,n_clusters_per_class=1,
                          n_informative=2,n_redundant=0,class_sep=1.0,flip_y=0)
    X = MinMaxScaler().fit_transform(X)
    Xp = np.random.uniform(size=(n_samples,2))
    lr = LogisticRegression().fit(X,y)
    yp = lr.predict(Xp)
    d = lr.decision_function(Xp)
    idx = ((d>.5) | (d<-.5))
    return Xp[idx],yp[idx]

class NoisyEstimator(BaseEstimator,ClassifierMixin):
    def __init__(self,estimator,PN,NP):
        self.estimator = estimator
        self.PN = PN
        self.NP = NP
        
    def fit(self,X,Y):
        Yn = create_noise(Y,self.PN,self.NP)
        print(np.unique(Y,return_counts=True))
        self.estimator.fit(X,Yn)
        return self
    
    def predict(self,X):
        return self.estimator.predict(X)