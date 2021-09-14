import numpy as np
from numpy import linalg as la

class VarianceEstimator:

    @staticmethod
    def regular_variance(X: np.array, SSR)->tuple:
        """Use SSR and x array to calculate different regular variance.

        Args:
            SSR (float): SSR
            x (np.array): Array of independent variables.
        Raises:
            Exception: [description]
        Returns:
            tuple: [description]
        """
        N = X.shape[0]
        K = X.shape[1]
        sigma = SSR / (N - K)  
        cov =  sigma*la.inv(X.T@X)
        se =  np.sqrt(cov.diagonal()).reshape(-1, 1)
        return sigma, cov, se
    @staticmethod
    def fe_variance(X:np.array, SSR)->tuple:
        pass

    @staticmethod
    def re_variance():
        pass


    
    def _perm(Q_T: np.array, A: np.array, t=0) -> np.array:
        """Takes a transformation matrix and performs the transformation on 
        the given vector or matrix.
        Args:
            Q_T (np.array): The transformation matrix. Needs to have the same
            dimensions as number of years a person is in the sample.
            
            A (np.array): The vector or matrix that is to be transformed. Has
            to be a 2d array.
        Returns:
            np.array: Returns the transformed vector or matrix.
        """
    # We can infer t from the shape of the transformation matrix.
        if t==0:
            t = Q_T.shape[1]

        # Initialize the numpy array
        Z = np.array([[]])
        Z = Z.reshape(0, A.shape[1])

        # Loop over the individuals, and permutate their values.
        for i in range(int(A.shape[0]/t)):
            Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
        return Z

class _BaseModel:
    """Instantiates base ols model.
    Args:
        x(np.array): Array of independent variable
    """
    def __init__(self, X:np.array, y:np.array):
        self._X = X
        self._y = y
        self._N = X.shape[0]
        self._K = X.shape[1]

    def _estimate_ols(self) -> tuple:
        self.b_hat = la.inv(self._X.T@self._X)@(self._X.T@self._y)
        R2, SSR, SST = self._metrics(self.b_hat)
        return self.b_hat, R2, SSR, SST

    def predict(self, X_new:np.array) -> np.array:
        """Uses estimated betas to predict an array X"""
        return X_new@self.b_hat
        
    def _metrics(self, b_hat) -> tuple:
        resid = self._y - self._X@b_hat
        SSR = np.sum(resid**2)
        SST = np.sum((self._y-self._y.mean())**2)
        R2 = 1 - (SSR / SST)
        return R2, SSR, SST


class PooledOLS(_BaseModel):

    _variance_estimator = VarianceEstimator()
    def fit(self) -> dict:
        b_hat, R2, SSR, SST =  self._estimate_ols()
        sigma, cov, se = self._variance_estimator.regular_variance(self._X, 
                                                                SSR = SSR)
        t_values = b_hat/se
        names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
        results = [b_hat, se, sigma, t_values, R2, cov]
        return dict(zip(names, results))

class FixedEffectsOLS(_BaseModel):
    def __init__(self, X:np.array, y:np.array):
        self._X = X
        self._y = y
        self._N = X.shape[0]
        self._K = X.shape[1]
        self._variance = VarianceEstimator()
    
    def fit(self):
        pass
