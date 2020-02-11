import numpy as np
from sklearn.base import ClassifierMixin,BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.utils import check_random_state
from sklearn.utils.fixes import parallel_helper, _joblib_parallel_args
from joblib import Parallel, delayed
from sklearn.neighbors.base import _get_weights

"""
Bagging and its subclasses has similar structure. fit() first initializes some stuff. 
Then modifies the X,Y through some type of randomization for each base.
"""

def kDN(X, Y, K=5, n_jobs=-1,weight='distance', **kwargs):
    knn = KNeighborsClassifier(n_neighbors=K, n_jobs=n_jobs, weights=weight).fit(X, Y)
    dist, kid = knn.kneighbors()  # (N,K) : ids & dist of nn's for every sample in X
    weights = _get_weights(dist, weight)
    if weights is None:
        weights = np.ones_like(kid)
    disagreement = Y[kid] != Y.reshape(-1, 1)
    return np.average(disagreement, axis=1, weights=weights)

def robust_kDN(X,Y,K=5,forest=None,n_jobs=1,weight='distance', random_state=None):
    if forest is None:
        forest = RandomForestClassifier(n_estimators=50,max_leaf_nodes=1000,n_jobs=n_jobs,random_state=random_state)
    forest = forest.fit(X,Y)
    Xs = forest.apply(X)
    knn = KNeighborsClassifier(n_neighbors=K,n_jobs=n_jobs,metric='hamming',algorithm='brute').fit(Xs,Y)
    dist, kid = knn.kneighbors()  # (N,K) : ids & dist of nn's for every sample in X
    weights = _get_weights(dist, weight)
    if weights is None:
        weights = np.ones_like(kid)
    disagreement = Y[kid] != Y.reshape(-1, 1)
    return np.average(disagreement, axis=1, weights=weights)

class FlipBase(BaseEstimator,ClassifierMixin):
    def __init__(self,estimator,flip_perc,sample_weight=None):
        super().__init__()
        self.estimator = estimator
        self.flip_perc = flip_perc
        self.sample_weight = sample_weight
    
    def fit(self,X,Y):
        rng = check_random_state(self.estimator.random_state)
        X,Y = noisify(X,Y,self.flip_perc,rng,sample_weight=self.sample_weight)
        self.estimator = self.estimator.fit(X, Y, )
        return self.estimator
    
    def predict(self,X):
        return self.estimator.predict(X)

    def apply(self,X,*args):
        return self.estimator.apply(X)

class LSBagging(BaggingClassifier):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 frac=.6):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=False, 
            bootstrap_features=False,
            oob_score=False,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.frac = frac
        
    def _validate_y(self, y):
        self.no_of_labels_ = len(np.unique(y))
        return super()._validate_y(y)   
        
    def _validate_estimator(self, default=DecisionTreeClassifier()):
        super()._validate_estimator()
        noise_max = (self.no_of_labels_-1)/self.no_of_labels_
        self.base_estimator_ = FlipBase(self.base_estimator_,self.frac*noise_max)

    def apply(self,X):
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           **_joblib_parallel_args(prefer="threads"))(
            delayed(parallel_helper)(tree, 'apply', X)
            for tree in self.estimators_)
        return np.array(results).T
 

class RobustLSB(LSBagging):
    def __init__(self,
        detector,
        base_estimator=None,
        n_estimators=10,
        warm_start=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        frac=.6,
        noise_model=None,
        K = 5):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            frac=frac)

        self.noise_model = noise_model
        self.detector = detector
        self.K = K
        
    def _validate_y(self, y):
        self.no_of_labels_ = len(np.unique(y))
        return super()._validate_y(y)
        
    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        super(LSBagging,self)._validate_estimator()
        noise_max = (self.no_of_labels_-1)/self.no_of_labels_
        self.base_estimator_ = FlipBase(self.base_estimator_,self.frac*noise_max,self.sample_weight_)
        
    def fit(self, X, Y, **kwargs):
        self.sample_weight_ = self.detector(X,Y,K=self.K,forest=self.noise_model,n_jobs=self.n_jobs) + .01
        self.sample_weight_ = self.sample_weight_/self.sample_weight_.sum()
        return super().fit(X,Y)

class WBBase(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator,replacement,sub_sample, sample_weight=None):
        super().__init__()
        self.estimator = estimator
        self.replacement = replacement
        self.sub_sample = sub_sample
        self.sample_weight = sample_weight

    def fit(self, X, Y):
        rng = check_random_state(self.estimator.random_state)
        to_sample = int(self.sub_sample*len(Y))
        target_idx = rng.choice(len(Y),size=to_sample,replace=self.replacement,p=self.sample_weight)
        X,Y = X[target_idx],Y[target_idx]
        self.estimator = self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

class WeightedBagging(BaggingClassifier):
    def __init__(self,
                 detector,
                 base_estimator=None,
                 n_estimators=100,
                 replacement=True,
                 sub_sample=1.0,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 K=5):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            bootstrap=False,
            bootstrap_features=False,
            verbose=verbose)
        self.detector = detector
        self.replacement = replacement
        self.sub_sample = sub_sample
        self.K = K

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        super()._validate_estimator()
        self.base_estimator_ = WBBase(self.base_estimator_, self.replacement,self.sub_sample, self.sample_weight_)

    def fit(self, X, Y, **kwargs):
        self.sample_weight_ = (1 - self.detector(X, Y, K=self.K,n_jobs=self.n_jobs)) + 1/(len(Y))
        self.sample_weight_ = self.sample_weight_ / self.sample_weight_.sum()
        return super().fit(X, Y)
    