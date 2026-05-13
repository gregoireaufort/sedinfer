import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(u):
    return np.log(u) - np.log1p(-u)

class BoxLogitTransform:
    """
    y in R^d  -> u = sigmoid(y) in (0,1)^d -> theta = low + (high-low)*u
    """
    def __init__(self, low, high, eps=1e-12):
        self.low = np.asarray(low, float)
        self.high = np.asarray(high, float)
        self.eps = float(eps)
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have same shape")
        if np.any(self.high <= self.low):
            raise ValueError("Need high > low elementwise")
        self.scale = self.high - self.low

    def y_to_theta(self, y):
        u = sigmoid(np.asarray(y, float))
        return self.low + self.scale * u

    def theta_to_y(self, theta):
        theta = np.asarray(theta, float)
        u = (theta - self.low) / self.scale
        u = np.clip(u, self.eps, 1.0 - self.eps)
        return logit(u)

    def log_abs_det_jac(self, y):
        y = np.asarray(y, float)
        u = sigmoid(y)
        u = np.clip(u, self.eps, 1.0 - self.eps)
        return np.sum(np.log(self.scale) + np.log(u) + np.log1p(-u))