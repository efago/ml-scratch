import numpy as np
from tree import DecisionTreeClassifier
from tree import DecisionTreeRegressor

from copy import deepcopy
from collections import Counter

from sklearn.datasets import load_breast_cancer, load_boston


class AdaBoost:
    """classifier using AdaBoost boosting algorithm
        expects binary targets of -1 and 1"""
    def __init__(
        self, 
        base_estimator= DecisionTreeClassifier(max_depth=1),
        n_estimators=50):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        # placeholder for boosted estimators
        self.estimators = []
        # estimator weights in majority vote predictions
        self.estimator_weights = np.empty(n_estimators)

    def fit(self, x, y):
        m = len(x)
        sample_weights = np.ones(m) / m

        for i in range(self.n_estimators):
            indices = np.random.choice(m, m, p=sample_weights)
            estimator = deepcopy(self.base_estimator)
            estimator.fit(x[indices], y[indices])
            predictions = estimator.predict(x)

            score = np.sum(sample_weights * y * predictions)
            exponential = -1 * score * y * predictions

            self.estimator_weights[i] = 0.5 * np.log((1 + score) / (1 - score))
            sample_weights *= np.exp(exponential)
            sample_weights /= np.sum(sample_weights)
            self.estimators.append(estimator)

    def predict(self, x):
        predictions = np.empty((self.n_estimators, len(x)))
        for i, estimator in enumerate(self.estimators):
            predictions[i] = estimator.predict(x)
           
        weighted_prediction = \
            self.estimator_weights.reshape(-1, 1) * predictions
        votes = (np.sum(weighted_prediction, axis=0) > 0) * 2 - 1

        return votes


class RandomForest:
    def __init__(self, estimator, n_estimators, criterion, max_depth,\
        max_leaves, max_features, min_sample_split):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.estimators = []        # placeholder for estimators
        self.tree_parameters = {
            'criterion': criterion,
            'max_depth': max_depth,
            'max_leaves': max_leaves,
            'max_features': max_features,
            'min_sample_split': min_sample_split
        }

    def fit(self, x, y):
        m = len(x)
        for _ in range(self.n_estimators):
            indices = np.random.choice(m, m)
            estimator = self.estimator(**self.tree_parameters)
            estimator.fit(x[indices], y[indices])
            self.estimators.append(estimator)

    def predict(self, x):
        predictions = np.empty((self.n_estimators, len(x)))
        for i, estimator in enumerate(self.estimators):
            predictions[i] = estimator.predict(x)

        return predictions


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, criterion='gini', max_depth=None,\
        max_leaves=None, max_features=None, min_sample_split=None):
        super().__init__(DecisionTreeClassifier, n_estimators, \
            criterion, max_depth, max_leaves, max_features, min_sample_split)

    def predict(self, x):
        predictions = super().predict(x)

        votes = np.empty(len(x))
        for i in range(predictions.shape[1]):
            vote = Counter(predictions[:, i]).most_common()[0][0]
            votes[i] = vote

        return votes


class RandomForestRegressor(RandomForest):
    def __init__(self, n_estimators=10, criterion='mse', max_depth=None,\
        max_leaves=None, max_features=None, min_sample_split=None):
        super().__init__(DecisionTreeRegressor, n_estimators, criterion, \
            max_depth, max_leaves, max_features, min_sample_split)

    def predict(self, x):
        predictions = super().predict(x)
        votes = np.mean(predictions, axis=0)

        return votes


def test_adaboost():
    x, y = load_breast_cancer(True)
    y = y*2 - 1

    model = AdaBoost(n_estimators=100)
    model.fit(x, y)
    predictions = model.predict(x)
    print(np.sum(predictions != y))

def test_random_forest_classifier():
    x, y = load_breast_cancer(True)

    model = RandomForestClassifier(n_estimators=20, max_depth=5, max_features=10)
    model.fit(x, y)
    predictions = model.predict(x)
    print(np.sum(predictions != y))

def test_random_forest_regressor():
    x, y = load_boston(True)

    model = RandomForestRegressor(n_estimators=20, max_depth=5, max_features=10)
    model.fit(x, y)
    predictions = model.predict(x)
    print(np.mean((predictions - y)**2))

def test_tree_classifier():
    x, y = load_breast_cancer(True)
    y = y*2 - 1

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(x, y)
    predictions = model.predict(x)
    print(np.sum(predictions != y))

def test_tree_regressor():
    x, y = load_boston(True)

    model = DecisionTreeRegressor(max_depth=5)
    model.fit(x, y)
    predictions = model.predict(x)
    print(np.mean((predictions - y)**2))
    

if __name__ == '__main__':
    #test_adaboost()
    #test_random_forest_classifier()
    #test_tree_classifier()
    test_random_forest_regressor()
    test_tree_regressor()


