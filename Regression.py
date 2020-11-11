import numpy as np
from ..utils.testing import is_symmetric_positive_definite, is_number


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        pseudo_inverse = np.linalg.inv(X.T @ X) @ X.T
        self.beta = np.dot(pseudo_inverse, y)

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)


class RidgeRegression:
    def __init__(self, alpha=1, fit_intercept=True):
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        A = self.alpha * np.eye(X.shape[1])
        pseudo_inverse = np.linalg.inv(X.T @ X + A) @ X.T
        self.beta = pseudo_inverse @ y

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)


class LogisticRegression:
    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.beta = None
        self.gamma = gamma
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        l_prev = np.inf
        self.beta = np.random.rand(X.shape[1])
        for _ in range(int(max_iter)):
            y_pred = sigmoid(np.dot(X, self.beta))
            loss = self._NLL(X, y, y_pred)
            if l_prev - loss < tol:
                return
            l_prev = loss
            self.beta -= lr * self._NLL_grad(X, y, y_pred)

    def _NLL(self, X, y, y_pred):
        N, M = X.shape
        beta, gamma = self.beta, self.gamma 
        order = 2 if self.penalty == "l2" else 1
        norm_beta = np.linalg.norm(beta, ord=order)
        
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        penalty = (gamma / 2) * norm_beta ** 2 if order == 2 else gamma * norm_beta
        return (penalty + nll) / N

    def _NLL_grad(self, X, y, y_pred):
        """Gradient of the penalized negative log likelihood wrt beta"""
        N, M = X.shape
        l1norm = lambda x: np.linalg.norm(x, 1)  # noqa: E731
        p, beta, gamma = self.penalty, self.beta, self.gamma
        d_penalty = gamma * beta if p == "l2" else gamma * np.sign(beta)
        return -(np.dot(y - y_pred, X) + d_penalty) / N

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(np.dot(X, self.beta))


class BayesianLinearRegressionUnknownVariance:
    def __init__(self, alpha=1, beta=2, b_mean=0, b_V=None, fit_intercept=True):
        # this is a placeholder until we know the dimensions of X
        b_V = 1.0 if b_V is None else b_V

        if isinstance(b_V, list):
            b_V = np.array(b_V)

        if isinstance(b_V, np.ndarray):
            if b_V.ndim == 1:
                b_V = np.diag(b_V)
            elif b_V.ndim == 2:
                fstr = "b_V must be symmetric positive definite"
                assert is_symmetric_positive_definite(b_V), fstr

        self.b_V = b_V
        self.beta = beta
        self.alpha = alpha
        self.b_mean = b_mean
        self.fit_intercept = fit_intercept
        self.posterior = {"mu": None, "cov": None}
        self.posterior_predictive = {"mu": None, "cov": None}

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape
        beta = self.beta
        self.X, self.y = X, y

        if is_number(self.b_V):
            self.b_V *= np.eye(M)

        if is_number(self.b_mean):
            self.b_mean *= np.ones(M)

        # sigma
        I = np.eye(N)  # noqa: E741
        a = y - np.dot(X, self.b_mean)
        b = np.linalg.inv(np.dot(X, self.b_V).dot(X.T) + I)
        c = y - np.dot(X, self.b_mean)

        shape = N + self.alpha
        sigma = (1 / shape) * (self.alpha * beta ** 2 + np.dot(a, b).dot(c))
        scale = sigma ** 2

        # b_sigma is the mode of the inverse gamma prior on sigma^2
        b_sigma = scale / (shape - 1)

        # mean
        b_V_inv = np.linalg.inv(self.b_V)
        L = np.linalg.inv(b_V_inv + X.T @ X)
        R = b_V_inv @ self.b_mean + X.T @ y

        mu = L @ R
        cov = L * b_sigma

        # posterior distribution for sigma^2 and c
        self.posterior = {
            "sigma**2": {"dist": "InvGamma", "shape": shape, "scale": scale},
            "b | sigma**2": {"dist": "Gaussian", "mu": mu, "cov": cov},
        }

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])  # noqa: E741
        mu = np.dot(X, self.posterior["b | sigma**2"]["mu"])
        cov = np.dot(X, self.posterior["b | sigma**2"]["cov"]).dot(X.T) + I

        # MAP estimate for y corresponds to the mean of the posterior
        # predictive
        self.posterior_predictive["mu"] = mu
        self.posterior_predictive["cov"] = cov
        return mu


class BayesianLinearRegressionKnownVariance:
    def __init__(self, b_mean=0, b_sigma=1, b_V=None, fit_intercept=True):
        # this is a placeholder until we know the dimensions of X
        b_V = 1.0 if b_V is None else b_V

        if isinstance(b_V, list):
            b_V = np.array(b_V)

        if isinstance(b_V, np.ndarray):
            if b_V.ndim == 1:
                b_V = np.diag(b_V)
            elif b_V.ndim == 2:
                fstr = "b_V must be symmetric positive definite"
                assert is_symmetric_positive_definite(b_V), fstr

        self.posterior = {}
        self.posterior_predictive = {}

        self.b_V = b_V
        self.b_mean = b_mean
        self.b_sigma = b_sigma
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape
        self.X, self.y = X, y

        if is_number(self.b_V):
            self.b_V *= np.eye(M)

        if is_number(self.b_mean):
            self.b_mean *= np.ones(M)

        b_V = self.b_V
        b_mean = self.b_mean
        b_sigma = self.b_sigma

        b_V_inv = np.linalg.inv(b_V)
        L = np.linalg.inv(b_V_inv + X.T @ X)
        R = b_V_inv @ b_mean + X.T @ y

        mu = L @ R
        cov = L * b_sigma ** 2

        # posterior distribution over b conditioned on b_sigma
        self.posterior["b"] = {"dist": "Gaussian", "mu": mu, "cov": cov}

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])  # noqa: E741
        mu = np.dot(X, self.posterior["b"]["mu"])
        cov = np.dot(X, self.posterior["b"]["cov"]).dot(X.T) + I

        # MAP estimate for y corresponds to the mean of the posterior
        # predictive distribution
        self.posterior_predictive = {"dist": "Gaussian", "mu": mu, "cov": cov}
        return mu


#######################################################################
#                                Utils                                #
#######################################################################


def sigmoid(x):
    """The logistic sigmoid function"""
    return 1 / (1 + np.exp(-x))
