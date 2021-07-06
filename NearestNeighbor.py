import multiprocessing
import numpy as np
from collections import Counter


class NearestNeighbor:

    def __init__(self, n_neighbors = 1, metric = 'l2', n_jobs = 1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, Y):
        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)
        assert X.ndim == 2 and np.squeeze(Y).ndim == 1
        self.x = X
        self.y = np.squeeze(Y)

    def predict(self, X):
        assert isinstance(X, np.ndarray)
        if self.n_jobs > 1:
            with multiprocessing.Pool(processes = self.n_jobs) as pool:
                predictions = pool.map(self.findNeighbor, X)
        else:
            predictions = []
            for instance in X:
                predictions.append(self.findNeighbor(instance))
        return predictions

    def findNeighbor(self, instance):

        knearest_y = np.array([-1] * self.n_neighbors)
        minimum_distance = np.array([np.inf] * self.n_neighbors)

        for i, x in enumerate(self.x):
            if self.metric == 'l2':
                distance = np.sqrt(np.sum(np.square(instance - x)))
            else:
                distance = np.sum(np.abs(instance - x))
            if distance < max(minimum_distance):
                argmax = np.argmax(minimum_distance)
                minimum_distance[argmax] = distance
                knearest_y[argmax] = self.y[i]

        nearest_y = Counter(knearest_y).most_common(1)[0][0]
        
        return nearest_y



