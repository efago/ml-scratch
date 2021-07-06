import numpy as np
from numpy.linalg import det, inv
import matplotlib.pyplot as plt

class GaussianMixin:
    def priors(self):
        class_priors = [np.sum(self.y == label) for label in self.classes]
        return np.array(class_priors) / len(self.y)

    def means(self):
        return np.array([np.mean(self.x[self.y == label], axis = 0) for label in self.classes])

    def covariances(self):
        num_classes = len(self.classes)
        covariance_matrix = np.zeros((num_classes, self.features, self.features))
        normalizers = np.zeros(num_classes)
        
        for i, label in enumerate(self.classes):
            x = self.x[self.y == label]
            for j in range(self.features):
                for k in range(j, self.features):
                    if x.ndim > 1:
                        covariance = np.mean(x[:, j] * x[:, k]) \
                            - self.means[i, j] * self.means[i, k]
                    else:
                        covariance = np.mean(x**2) - self.means[i]**2

                    covariance_matrix[i, j, k] = covariance_matrix[i, k, j] = covariance
            
            normalizer = (2 * np.pi)**(self.features / 2) * np.sqrt(det(covariance_matrix[i]))
            normalizers[i] = 1 / normalizer
            #added a constant to avoid singularity
            covariance_matrix += np.diag([0.0001]*self.features)

        return covariance_matrix, normalizers

    def likelihoods(self, x):
        class_likelihoods = []

        for i in range(len(self.classes)):
            dot_products = np.dot(x - self.means[i], inv(self.covariances[i]))
            dot_products = np.dot( dot_products, (x - self.means[i]).T)
            likelihood = self.normalizers[i] * np.exp(-0.5 * dot_products)

            class_likelihoods.append(likelihood)

        return class_likelihoods

    def plot_distribution(self, feature, label_index):
        """plot gaussian distribution for single feature
        """
        mean = self.means[label_index, feature]
        variance = self.covariances[label_index, feature, feature]
        normalizer = 1 / np.sqrt(2 * np.pi * variance)
        density_function = lambda x: normalizer * np.exp((x - mean)**2 / (-2 * variance))

        x = np.linspace(mean - 3 * np.sqrt(variance), mean + 3 * np.sqrt(variance), 40)
        y = density_function(x)
        y_label = self.classes[label_index]
        plt.plot(x, y)
        plt.ylabel(f'Class {y_label}')
        plt.title(f'Density function of class {y_label} for feature {feature}')
        plt.show()


class GenerativeModel(GaussianMixin):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.classes = np.unique(y)
        self.features = x.shape[1] if x.ndim > 1 else 1
        self.means = self.means()
        self.covariances, self.normalizers = self.covariances() 
        self.priors = self.priors()

    def predict(self, x):
        """predict using bayes rule - argmax(likelihood * prior)
        """
        likelihoods = [self.likelihoods(instance) for instance in x]
        posteriors = self.priors * np.squeeze(likelihoods)
        predictions = self.classes[np.argmax(posteriors, axis = 1)]
        
        return predictions

    def __repr__(self):
        return f'Gaussian generative model(features: {self.features},\
            classes: {self.classes}'

    def __str__(self):
        return f'Gaussian generative model'
