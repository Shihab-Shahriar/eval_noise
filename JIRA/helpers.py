import numpy as np
from scipy.stats import mode
from sklearn.base import ClassifierMixin,clone,BaseEstimator,clone
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.utils.extmath import weighted_mode
from sklearn.neighbors.base import _get_weights
from scipy.spatial.distance import cdist,hamming

__all__ = ['RobustKNN']

class RobustKNN(BaseEstimator,ClassifierMixin):
    def __init__(self,K=5,forest=None,n_estimators=1000,random_state=None,n_jobs=None,method='simple'):
        self.K = K
        self.forest = RandomForestClassifier(n_estimators=n_estimators,max_leaf_nodes=1000,n_jobs=n_jobs,random_state=random_state)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.method = method
    
    def fit(self,X,Y):
        if self.fit_nm: 
            self.forest = self.forest.fit(X, Y)
        self.data = (X,Y)
        return self

    def _pairwise_distance_weighted(self, train_X, test_X):
        out_shape = (test_X.shape[0], train_X.shape[0])
        mat = np.zeros(out_shape,dtype='float32')
        temp = np.zeros(out_shape,dtype='float32')
        to_add = np.zeros(out_shape,dtype='float32')
        ones = np.ones(out_shape,dtype='float32')
        for tree in self.forest.estimators_:
            train_leaves = tree.apply(train_X)
            test_leaves = tree.apply(test_X)
            match = test_leaves.reshape(-1, 1) == train_leaves.reshape(1, -1)  # Samples w/ same leaf as mine:mates
            no_of_mates = match.sum(axis=1, dtype='float')  # No of My Leaf mates
            np.multiply(match, no_of_mates.reshape(-1, 1),out=temp)  # assigning weight to each leaf mate, proportional to no of mates
            to_add.fill(0)
            np.divide(ones, temp, out=to_add, where=temp != 0)  # Now making that inversely proportional
            assert np.allclose(to_add.sum(axis=1), 1)
            assert match.shape == (len(test_X), len(train_X)) == to_add.shape == temp.shape
            assert no_of_mates.shape == (len(test_X),)
            np.add(mat, to_add, out=mat)
        return 1 - mat / len(self.forest.estimators_)

    def _pairwise_distance_simple(self, train_X, test_X):
        train_leaves = self.forest.apply(train_X) # (train_X,n_estimators)
        test_leaves = self.forest.apply(test_X)   # (test_X,n_estimators)
        dist = cdist(test_leaves,train_leaves,metric='hamming')
        assert dist.shape==(len(test_X),len(train_X))
        return dist

    def pairwise_distance(self,train_X,test_X):
        if self.method=='simple':
            return self._pairwise_distance_simple(train_X,test_X)
        elif self.method=='weighted':
            return self._pairwise_distance_weighted(train_X,test_X)
        raise Exception("method not recognized")

    
    def predict(self,X):
        train_X,train_Y = self.data
        dist = self.pairwise_distance(train_X,X)
        assert np.all(dist>=0)
        idx = np.argsort(dist,axis=1)
        nn_idx = idx[:,:self.K]
        nn_dist = dist[np.arange(len(X))[:,None],nn_idx]
        nn_labels = train_Y[nn_idx]
        weights = _get_weights(nn_dist,'distance')
        a,_ = weighted_mode(nn_labels,weights,axis=1)
        return a.reshape(-1)