import numpy as np
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def noisify(Y:np.ndarray,frac:float,random_state=None,sample_weight=None,index=False): #This target_mask may be not needed
    random_state = check_random_state(random_state)
    nns = int(len(Y)*frac)
    labels = np.unique(Y)
    target_idx = random_state.choice(len(Y),size=nns,replace=False,p=sample_weight)
    #print(target_idx[:3])
    target_mask = np.full(Y.shape,0,dtype=np.bool)
    target_mask[target_idx] = 1
    w = Y.copy()
    mask = target_mask.copy()
    while True:
        left = mask.sum()
        #print(left)
        if left==0:break
        new_labels = random_state.choice(labels,size=left)
        w[mask] = new_labels
        mask = mask & (w==Y)
    assert (w[target_idx]==Y[target_idx]).sum()==0
    assert (w[~target_mask]==Y[~target_mask]).sum()==len(Y)-nns
    if index: return w,target_mask
    return w

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

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, power_transform
from sklearn.utils import shuffle


file = """AvgCyclomatic, AvgCyclomaticModified, AvgCyclomaticStrict, AvgEssential, AvgLine, AvgLineBlank, AvgLineCode, AvgLineComment, CountDeclClass, CountDeclClassMethod, CountDeclClassVariable, CountDeclFunction, CountDeclInstanceMethod,
CountDeclInstanceVariable, CountDeclMethod, CountDeclMethodDefault, CountDeclMethodPrivate, CountDeclMethodProtected,
CountDeclMethodPublic, CountLine, CountLineBlank, CountLineCode, CountLineCodeDecl, CountLineCodeExe, CountLineComment, CountSemicolon, CountStmt, CountStmtDecl, CountStmtExe, MaxCyclomatic, MaxCyclomaticModified, MaxCyclomaticStrict, RatioCommentToCode, SumCyclomatic, SumCyclomaticModified, SumCyclomaticStrict, SumEssential"""
cls = """CountClassBase, CountClassCoupled, CountClassDerived, MaxInheritanceTree, PercentLackOfCohesion"""
meth_prefix = ["CountInput","CountOutput","CountPath","MaxNesting"]

file_metrics = [c.strip() for c in file.split(',')]
cls_metrics = [c.strip() for c in cls.split(',')]
meth_metrics = ['CountInput_Max', 'CountInput_Mean', 'CountInput_Min','CountOutput_Max','CountOutput_Mean',
 'CountOutput_Min','CountPath_Max','CountPath_Mean','CountPath_Min','MaxNesting_Max','MaxNesting_Mean','MaxNesting_Min']
code_metrics = set(file_metrics) | set(cls_metrics) | set(meth_metrics)
process_metrics = ["COMM","Added_lines","Del_lines","ADEV","DDEV"]
own_metrics = ["OWN_LINE","OWN_COMMIT","MINOR_LINE","MINOR_COMMIT","MAJOR_COMMIT","MAJOR_LINE"]
all_metrics = set(code_metrics) | set(own_metrics) | set(process_metrics)

def read_jira(file,stats=True):
    df = pd.read_csv("JIRA/"+file)
    df.drop(columns=["File",'HeuBugCount','RealBugCount'],inplace=True)
    df = shuffle(df)
    X = df[all_metrics].values.astype('float64')
    y_noisy = df.HeuBug.values.astype('int8')
    y_real = df.RealBug.values.astype('int8')
    X = np.log1p(X)                      #INFORMATION LEAK, could also use power_transform
    #X = MinMaxScaler().fit_transform(X) #Use of this two transformer needs to be looked at
    assert y_noisy.sum()<len(y_noisy)*.5   #Ensure 1 is bug
    if stats:
        noise = (df.HeuBug!=df.RealBug).sum()/len(df)
        imb = np.unique(y_noisy,return_counts=True)[1]
        print(f"{file} noise:{noise:.3f}, imb:{imb.max()/imb.min():.3f},{imb.min()},{imb.max()}, Shape:{X.shape}")
        
    return X,y_noisy,y_real

