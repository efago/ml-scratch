import numpy as np
from numpy.linalg import inv


class Regression:
    def __init__(self, x, y, lambd = 0):
        self.x = x if x.ndim > 1 else x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        self.coefficients, self.intercept = LineFitter().get_w(self.x, self.y, lambd)

    @property
    def mse(self):
        y_predictions = np.dot(self.x, self.coefficients) + self.intercept
        return np.mean((self.y - y_predictions)**2)

    def predict(self, x):
        x = x if x.ndim > 1 else x.reshape(-1, 1)
        return np.dot(x, self.coefficients) + self.intercept

class LineFitter:
    """computes coefficients (intercept and slope) that minimize mse
    """
    def get_w(self, x , y, lambda_):
        x_tilde = np.concatenate([np.ones((x.shape[0], 1)), x], axis = 1)
        dot_product1 = inv(np.dot(x_tilde.T, x_tilde) + \
            np.diag([lambda_] * x_tilde.shape[1]))
        dot_product2 = np.dot(x_tilde.T, y)
        w = np.dot(dot_product1, dot_product2)

        return w[1:], w[0]

class LogisticRegression(Regression):
    pass

class RidgeRegression(Regression):
    def __init__(self, x, y, lambda_):
        self.lambda_ = lambda_
        super().__init__(x, y, lambda_)

    def __repr__(self):
        return f'Ridge Regression(lambda: {self.lambda_}, w: {self.coefficients} \
            b: {self.intercept})'

class LinearRegression(Regression):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __repr__(self):
        return f'Least Squares Regression(w: {self.coefficients} \
            b: {self.intercept})'