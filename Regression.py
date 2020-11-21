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


class Softmax(LayerBase):
    def __init__(self, dim=-1, optimizer=None):
        r"""
        A softmax nonlinearity layer.
        Notes
        -----
        This is implemented as a layer rather than an activation primarily
        because it requires retaining the layer input in order to compute the
        softmax gradients properly. In other words, in contrast to other
        simple activations, the softmax function and its gradient are not
        computed elementwise, and thus are more easily expressed as a layer.
        The softmax function computes:
        .. math::
            y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
        where :math:`x_i` is the `i` th element of input example **x**.
        Parameters
        ----------
        dim: int
            The dimension in `X` along which the softmax will be computed.
            Default is -1.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None. Unused for this layer.
        """  # noqa: E501
        super().__init__(optimizer)

        self.dim = dim
        self.n_in = None
        self.is_initialized = False

    def _init_params(self):
        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "SoftmaxLayer",
            "n_in": self.n_in,
            "n_out": self.n_in,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Layer input, representing the `n_in`-dimensional features for a
            minibatch of `n_ex` examples.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.
        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            Layer output for each of the `n_ex` examples.
        """
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = self._fwd(X)

        if retain_derived:
            self.X.append(X)

        return Y

    def _fwd(self, X):
        """Actual computation of softmax forward pass"""
        # center data to avoid overflow
        e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))
        return e_X / e_X.sum(axis=self.dim, keepdims=True)

    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs.
        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)` or list of arrays
            The gradient(s) of the loss wrt. the layer output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.
        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer input `X`.
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx = self._bwd(dy, x)
            dX.append(dx)

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """
        Actual computation of the gradient of the loss wrt. the input X.
        The Jacobian, J, of the softmax for input x = [x1, ..., xn] is:
            J[i, j] =
                softmax(x_i)  * (1 - softmax(x_j))  if i = j
                -softmax(x_i) * softmax(x_j)        if i != j
            where
                x_n is input example n (ie., the n'th row in X)
        """
        dX = []
        for dy, x in zip(dLdy, X):
            dxi = []
            for dyi, xi in zip(*np.atleast_2d(dy, x)):
                yi = self._fwd(xi.reshape(1, -1)).reshape(-1, 1)
                dyidxi = np.diagflat(yi) - yi @ yi.T  # jacobian wrt. input sample xi
                dxi.append(dyi @ dyidxi)
            dX.append(dxi)
        return np.array(dX).reshape(*X.shape)

#######################################################################
#                                Utils                                #
#######################################################################

def sigmoid(x):
    """The logistic sigmoid function"""
    return 1 / (1 + np.exp(-x))
