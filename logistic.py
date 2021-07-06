import numpy as np
from multiprocessing import Pool


class LogisticRegression:
    def __init__(self, gamma = 0.01, max_iteration = 1000, verbose = False):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.verbose = verbose

    def fit(self, x, y, batch_size = None):
        self.x = x if x.ndim > 1 else x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        self._w = self._get_w(batch_size)
        self.coefficients, self.intercept = self._w[1:], self._w[0]

    def predict(self, x):
        probabilities = self.predict_probabilities(x)
        return (probabilities >= 0.5) * 2 - 1

    def predict_probabilities(self, x):
        x = x if x.ndim > 1 else x.reshape(-1, 1)
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis = 1)
        y = np.ones((x.shape[0], 1))
        return self._sigmoid(x, y, self._w)

    def _sigmoid(self, x, y, w):
        return 1 / (1 + np.exp(-y * np.dot(x, w)))

    def _get_w(self, batch_size):
        sample_size = self.x.shape[0]
        w = np.random.randn(self.x.shape[1] + 1, 1)
        x = np.concatenate([np.ones((sample_size, 1)), self.x], axis = 1)
        if not batch_size:
            batch_size = sample_size #set batch_size to total samples if None
        
        for i in range(self.max_iteration):

            if self.verbose and i % 100 == 0:
                loss = -1 * np.sum(np.log(self._sigmoid(x, self.y, w)))
                print(f'Iteration {i}: loss {loss}')
            
            #mini batch gradient descent when batch_size is input
            batches = sample_size // batch_size
            for batch in range(batches):
                batch_slice = slice(batch_size * batch, (batch + 1) * batch_size)
                x_batch = x[batch_slice]
                y_batch = self.y[batch_slice]
    
                dw = np.sum(y_batch * x_batch * self._sigmoid(x_batch, -y_batch, w), axis = 0, keepdims = True)
                w += self.gamma * dw.T

        return w