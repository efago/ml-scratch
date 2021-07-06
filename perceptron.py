import numpy as np
from numpy.linalg import norm


class Perceptron:
    def __init__(self, kernel='quadratic', degree=1, s=1, max_iteration=200):
        assert kernel in ['quadratic', 'rbf'], kernel
        self.kernel = kernel
        self.degree = degree
        self.s = s              # scaling factor in rbf kernel
        self.max_iteration = max_iteration

        
    def fit(self, x, y):
        """fits a classifier to linearly separable data in 'self.degree' dimensions
            expects y to represent labels (0, 1, .... , n_classes)"""
        m, n = x.shape
        n_classes = len(np.unique(y))
        y = y.astype(np.int)

        self.kernel_matrix = self._get_kernel_matrix(x)    
        self.alpha = np.zeros((n_classes, m))
        self.b = np.zeros(n_classes)
        self.x = x
        correct = 0     # counter for number of correctly classified samples
        iteration = 0   # counter for passes through the data

        while iteration < self.max_iteration and correct < m:
            indices = np.random.permutation(m)

            for i in indices:
                predictions = self.alpha * self.kernel_matrix[i] + self.b
                prediction = np.argmax(predictions)

                if y[i] != prediction:
                    self.alpha[y[i], i] += 1
                    self.b[y[i]] += 1
                    self.alpha[prediction, i] -= 1
                    self.b[prediction] -= 1
                    correct = 0
                else:
                    correct += 1

            iteration += 1


    def predict(self, x):
        """expects x to be of shape (instances, features)
        y = np.dot(x, self.w.T) + self.b
        predictions = np.argmax(y, axis=1)"""
        

        return predictions

    def _get_kernel_matrix(self, x):
        """returns matrix with ij entry equal to the result of 
            kernel function computed for x_i and x_j"""
        m = len(x)
        kernel_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                if self.kernel == 'quadratic':
                    kernel_result = np.dot(x[i], x[j])
                    if self.degree > 1:
                        kernel_result = (1 + kernel_result)**self.degree
                else:
                    kernel_result = np.exp(-1 * norm(x[i] - x[j])**2 / self.s**2)
                kernel_matrix[i, j] = kernel_matrix[j, i] = kernel_result

        return kernel_matrix



if __name__ == '__main__':
    #data = np.loadtxt('perceptron_at_work/data_1.txt')
    #data = np.loadtxt('multiclass/data_3.txt')
    data = np.loadtxt('../week7/kernel/data1.txt')
    n,d = data.shape
    x = data[:,0:2]
    y = data[:,2]
    y[y == -1] = 0
    
    model = Perceptron(degree=2)
    model.fit(x, y)
    predictions = model.predict(x)
    print(np.all(predictions == y), sum(predictions != y))