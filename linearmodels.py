import numpy as np
from numpy import linalg as la

class Covariance:
    def __init__(self, exog):
        self.exog = exog
        self.N = exog.shape[0]
        self.K = exog.shape[1]

    def cov(self, SSR, type:str = 'standard')->tuple:
        """Use SSR and x array to calculate different regular variance.

        Args:
            SSR (float): SSR
            x (np.array): Array of independent variables.
        Raises:
            Exception: [description]
        Returns:
            tuple: [description]
        """

        if type == 'standard':
            sigma, cov, se = self.standard(SSR)
        elif type == 'fe':
            pass
        
        return sigma, cov, se
    
    def standard(self, SSR):
        sigma = SSR / (self.N - self.K)  
        cov =  sigma*la.inv(self.exog.T@self.exog)
        se =  np.sqrt(cov.diagonal()).reshape(-1, 1)
        return sigma, cov, se
        
class Transform:
    def _perm(self, Q_T: np.array, A: np.array, t=0) -> np.array:
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

class BaseModel(object):
    """Instantiates base-model.
    Args:
        exog(np.array): NxK Array of exogenous variables
        dependent(np.array): Kx1 Array of the dependent variable
    """
    def __init__(self, exog:np.array, dependent:np.array):
        #TODO: Check rank condition is satisfied
        self.exog = exog
        self.dependent= dependent
        self.N = exog.shape[0]
        self.K = exog.shape[1]
        self.transform = Transform()
        self.cov = Covariance(self.exog)

    def fit(self, cov_method:str = 'standard', transform = None) -> tuple:
        """
        Add docstring...
        """
        self.b_hat = self._ols()
        SSR = self._SSR()
        SST = self._SST()
        R2 = self._R2(SSR, SST)
        sigma, cov, se = self.cov.cov(SSR = SSR, type=cov_method)
        t_values = self.b_hat / se
        names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
        results = [self.b_hat, se, sigma, t_values, R2, cov]
        return dict(zip(names, results)) 

    def predict(self, X_new:np.array) -> np.array:
        """Uses estimated betas to predict an array X"""
        return X_new@self.b_hat
    
    def _ols(self):
        """
        Estimates OLS for various models. Data can be transformer.
        Math: (X'X)^{-1}(X'y)
        """
        XX = la.inv(self.exog.T@self.exog)
        Xy = self.exog.T@self.dependent
        return XX@Xy

    def _SSR(self) -> float:
        resid = self.dependent - self.exog@self.b_hat
        return np.sum(resid**2)

    def _SST(self)-> float:
        return np.sum((self.dependent-self.dependent.mean())**2)

    @staticmethod
    def _R2(SSR, SST)-> float:
        return 1 - (SSR / SST)
  

class OLS(BaseModel):
    pass
