import numpy as np
from numpy.linalg import det, inv
from scipy.stats import multivariate_normal


class Cluster:
    def __init__(self, k, initializer = 'random'):
        assert isinstance(k, int) and k > 1, k
        assert initializer in ['random', 'kmeans++'], initializer
        self.k = k
        self.initializer = initializer

    def initialize_means(self, x):
        samples = len(x)
        if self.initializer == 'random':
            means = x[np.random.choice(samples, self.k, replace = False)]
        else:
            means = [x[np.random.choice(samples, 1)[0]]]
            for _ in range(1, self.k):
                distances = []
                for mean in means:
                    distances.append(np.sum(np.square(x - mean), axis = 1))
                
                distances = np.min(distances, axis = 0)
                distribution = distances / np.sum(distances)

                mean = np.random.choice(samples, 1, p = distribution)[0]
                means.append(x[mean])
        
        return means


class GaussianMixture(Cluster):
    def __init__(self, k, initializer='random'):
        super().__init__(k, initializer=initializer)

    def fit(self, x):
        """fit the data into clusters using expectation maximization
        """
        means = self.initialize_means(x)
        covariances = np.array([np.cov(x, rowvar = False)] * self.k)
        priors = np.full((self.k, 1),  1 / self.k)
        clusters = np.zeros(len(x))

        while True:
            likelihoods = np.zeros((self.k, len(x)))
            for i, args in enumerate(zip(means, covariances)):
                likelihoods[i] = multivariate_normal.pdf(x, *args)

            posteriors = likelihoods * priors
            posteriors /= (np.sum(posteriors, axis = 0, keepdims = True))

            new_clusters = np.argmax(posteriors, axis = 0)
            if np.all(new_clusters == clusters):
                break
            else:
                clusters = new_clusters
                priors = np.mean(posteriors, axis = 1, keepdims = True)
                for i in range(self.k):
                    means[i] = np.mean(posteriors[i].reshape(-1, 1) * x, axis = 0) / priors[i]
                    covariances[i] = np.dot((posteriors[i].reshape(-1, 1) * (x - means[i])).T, (x - means[i]))\
                        / (priors[i] * len(x))
        
        self.means = means
        self.priors = priors
        self.covariances = covariances

    def __repr__(self):
        return f'Gaussian Mixture: (initializer, {self.initializer}),\
            (k, {self.k})'


class KMeans(Cluster):
    def __init__(self, k, initializer = 'random'):
        super().__init__(k, initializer)

    def fit( self, x):
        """fit the data into clusters using Lloyd's algorithm
        """
        means = self.initialize_means(x)
        samples = len(x)    
        clusters = np.zeros(samples)
        while True:
            new_clusters = np.zeros(samples)

            for i in range(samples):
                new_clusters[i] = np.argmin(np.sum(np.square(x[i] - means), axis = 1))

            if np.all(new_clusters == clusters):
                break
            else:
                for i in range(len(means)):
                    means[i] = np.mean(x[new_clusters == i], axis = 0)
                clusters = new_clusters

        self.means = means
        self.clusters = clusters

    def predict(self, x):
        """predicts for a single instance of x
        """
        return np.argmin(np.sum(np.square(x - self.means), axis = 1))

    def __repr__(self):
        return f'KMeans: (initializer, {self.initializer}), \
            (k, {self.k})'


if __name__ == '__main__':
    data = []
    clusters = np.array([0]*100 + [1]*100 + [2]*100)
    classes = [
        [0, (1,1), [[2, 0.5],[0.5, 1]]], 
        [1, (7, 12), [[3, 3],[3,10]]], 
        [2, (1, 10), [[2, 2],[2,10]]]]
    for parameters in classes:
        data.append(np.random.multivariate_normal(parameters[1], parameters[2], 100))
    
    data = np.array(data).reshape(-1, 2)
    gaussian_mixtures = GaussianMixture(3, 'kmeans++')
    gaussian_mixtures.fit(data)
    print(gaussian_mixtures.means)
    print(gaussian_mixtures.priors)