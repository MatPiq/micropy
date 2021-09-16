import numpy as np
from numpy import linalg as la
from IPython.display import display, Latex

class Covariance:
    def __init__(self, exog):
        self.exog = exog
        self.N = exog.shape[0]
        self.K = exog.shape[1]
 
    def olscov(self, resid, covmethod:str='standard'):
        """Calculates normal standard errors"""
        
        if covmethod == 'standard':
            sigma = np.sum(resid**2) / (self.N - self.K)  
            cov =  sigma*la.inv(self.exog.T@self.exog)
            se =  np.sqrt(cov.diagonal()).reshape(-1, 1)
            return sigma, cov, se
        elif covmethod == 'robust':
            #TODO:Does not work properly
            O = np.diag(resid@resid.T)
            XOX = self.exog.T@O@self.exog
            XX = la.inv(self.exog.T@self.exog)
            cov = XX@XOX@XX
            sd = np.sqrt(cov.diagonal()).reshape(-1,1)
            return None, cov, sd

    def fecov(self, resid, covmethod:str='standard'):
        pass

    def fdcov(self, resid, covmethod: str='standard'):
        pass

    def recov(self, resid, covmethod: str='standard'):
        pass

            
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
    """
        Instantiates base-model. Should not be called directly.
    Args:
        exog(np.array): NxK Array of exogenous variables
        dependent(np.array): Kx1 Array of the dependent variable
    """
    def __init__(self, exog:np.array, dependent:np.array):
        #TODO: Check rank condition is satisfied
        self._exog = exog
        self._dependent= dependent
        self._N = exog.shape[0]
        self._K = exog.shape[1]
        self._cov = Covariance(exog)

    def fit(self) -> tuple:
        """
        Add docstring...
        """
        self._b_hat = self._estimate()
        self._resid = self._resid()
        self._SSR = self._SSR()
        self._SST = self._SST()
        self._R2 = self._R2()

    def predict(self, X_new:np.array) -> np.array:
        """Uses estimated betas to predict an array X"""
        return X_new@self._b_hat
    
    def _estimate(self):
        """
        Estimates OLS for various models. Data can be transformed.
        Math: (X'X)^{-1}(X'y)
        """
        XX = la.inv(self._exog.T@self._exog)
        Xy = self._exog.T@self._dependent
        return XX@Xy
    
    def _resid(self) -> float:
        return self._dependent - self._exog@self._b_hat

    def _SSR(self) -> float:
        return np.sum(self._resid**2)

    def _SST(self)-> float:
        return np.sum((self._dependent-self._dependent.mean())**2)

    def _R2(self)-> float:
        return 1 - (self._SSR / self._SST)
  
class OLS(BaseModel):

    def fit(self, cov_method:str = 'standard'):
        super().fit()
        sigma, cov, se = self._cov.olscov(self._resid, cov_method)
        t_values = self._b_hat / se
        names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
        values = [self._b_hat, se, sigma, t_values, self._R2, cov]
        self.results = dict(zip(names, values)) 
        return self.results
        
    #@staticmethod
    def printmodel(self):
        """
        Only works in iPython. Prints how beta and variance is estimates.
        """
        #Beta
        display(Latex('$\hat{ \\beta } = (\mathbf{X\'X})(\mathbf{X\'y})$'))
        #Variance
        #TODO:add

class FixedEffectsOLS(BaseModel):
    
    def __init__(self, exog, dependent):
        super().__init__(exog, dependent)
        self.transformer = Transform()