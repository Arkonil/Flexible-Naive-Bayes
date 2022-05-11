import numpy as np

class GaussianNB:
    def __init__(self):
        self.init_params()

    def init_params(self):
        self.p = 0
        self.n_classes = 0
        self.counts = None
        self.mu = None
        self.sigma = None

    def train(self, X: np.ndarray, y: np.ndarray):
        self.init_params()

        _, self.p = X.shape
        self.n_classes = len(set(y))

        _, self.counts = np.unique(y, return_counts=True)

        self.mu = np.zeros((self.p, self.n_classes))
        self.sigma = np.zeros((self.p, self.n_classes))

        for i in range(self.n_classes):
            self.mu[:, i] = X[y == i, :].mean(axis=0).flatten()
            self.sigma[:, i] = X[y == i, :].std(axis=0).flatten()


    def predict(self, X: np.ndarray):
        y_prob = self.predict_prob(X)
        y_pred = y_prob.argmax(axis=1)
        return y_pred
        
    def predict_prob(self, X: np.ndarray):
        X = X.reshape(-1, self.p)
        m, p = X.shape

        y_prob = np.zeros((m, self.n_classes))

        for i in range(m):
            x = X[i, :].reshape(p, 1)
            M = -(((x - self.mu) / self.sigma) ** 2)
            M = np.exp(M.sum(axis=0))
            M /= self.sigma.prod(axis=0)
            M *= self.counts
            M /= M.sum()
            y_prob[i, :] = M

        return y_prob

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)