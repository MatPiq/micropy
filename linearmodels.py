import numpy as np
from IPython.display import Latex, display
from numpy import linalg as la


class VarianceEstimator:
    
    @staticmethod
    def ols(exog, resid, covmethod:str='standard'):
        """Calculates normal standard errors"""
        N = exog.shape[0]
        K = exog.shape[1]
        
        if covmethod == 'standard':
            sigma = resid.T@resid / (N - K)  
            cov =  sigma*la.inv(exog.T@exog)
            se =  np.sqrt(cov.diagonal()).reshape(-1, 1)
            return sigma, cov, se
        
        elif covmethod == 'robust':
            #TODO:Does not work properly
            O = np.diag(resid@resid.T)
            XOX = exog.T@O@exog
            XX = la.inv(exog.T@exog)
            cov = XX@XOX@XX
            sd = np.sqrt(cov.diagonal()).reshape(-1,1)
            return None, cov, sd
    
    @staticmethod
    def fe(exog, resid, t, covmethod:str='standard'):
        N = exog.shape[0] / t
        K = exog.shape[1] 

        if covmethod == 'standard':
            sigma = (resid.T@resid) / (N*(t-1)-K)
            cov =  sigma*la.inv(exog.T@exog)
            se =  np.sqrt(cov.diagonal()).reshape(-1, 1)
            return sigma, cov, se
    
    @staticmethod
    def fd(exog, resid, covmethod: str='standard'):
        N = exog.shape[0] 
        K = exog.shape[1] 

        if covmethod == 'standard':
            sigma = (resid.T@resid) / (N-K)
            cov =  sigma*la.inv(exog.T@exog)
            se =  np.sqrt(cov.diagonal()).reshape(-1, 1)
            return sigma, cov, se
    
    @staticmethod
    def re(resid, covmethod: str='standard'):
        pass

            
class Transform:
    
    def _perm(self, Q_T, A:np.array, t=0) -> np.array:
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
        
        # Initialize the numpy array
        Z = np.array([[]])
        Z = Z.reshape(0, A.shape[1])

        # Loop over the individuals, and permutate their values.
        for i in range(int(A.shape[0]/t)):
            Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
        return Z
    
    def fd_trans(self, A:np.array, t=0):
        D = np.zeros((t-1, t))
        np.fill_diagonal(np.flip(D), 1)
        np.fill_diagonal(D, -1)
        
        return self._perm(D, A, t)
    
    def fe_trans(self, A:np.array, t=0):
        Q_T = np.identity(t) - np.ones((t,t)) / t
        
        return self._perm(Q_T, A, t)
        
        

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
        self._variance = VarianceEstimator()
        self._transformer = Transform()

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
    
    def _estimate(self) -> np.array:
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
        return self._resid.T@self._resid

    def _SST(self)-> float:
        return np.sum((self._dependent-self._dependent.mean())**2)

    def _R2(self)-> float:
        return 1 - (self._SSR / self._SST)
  
class OLS(BaseModel):

    def fit(self, cov_method:str = 'standard'):
        super().fit()
        sigma, cov, se = self._variance.ols(self._exog, self._resid, cov_method)
        t_values = self._b_hat / se
        names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
        values = [self._b_hat, se, sigma, t_values, self._R2, cov]
        self.results = dict(zip(names, values)) 
        return self.results
        
    #@staticmethod
    def printmodel(self):
        """[summary]
        """
        #Beta
        display(Latex('$\hat{ \\beta } = (\mathbf{X\'X})(\mathbf{X\'y})$'))
        #Variance
        #TODO:add

class FixedEffectsOLS(BaseModel):
    
    def __init__(self, exog, dependent, t:int=1):
        super().__init__(exog, dependent)
        #Transform input by de-meaning
        self._t = t
        self._exog = self._transformer.fe_trans(exog, t)
        self._dependent = self._transformer.fe_trans(dependent, t)

    def fit(self, cov_method:str = 'standard'):
        """[summary]

        Args:
            cov_method (str, optional): [description]. Defaults to 'standard'.

        Returns:
            [type]: [description]
        """
        super().fit()
        sigma, cov, se = self._variance.fe(self._exog, self._resid,
                                             self._t, cov_method)
        t_values = self._b_hat / se
        names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
        values = [self._b_hat, se, sigma, t_values, self._R2, cov]
        self.results = dict(zip(names, values)) 
        return self.results
    
class FirstDifferenceOLS(BaseModel):
    
    def __init__(self, exog, dependent, t:int=1):
        super().__init__(exog, dependent)
        #Transform input by de-meaning
        self._t = t
        self._exog = self._transformer.fd_trans(exog, t)
        self._dependent = self._transformer.fd_trans(dependent, t)

    def fit(self, cov_method:str = 'standard'):
        """[summary]

        Args:
            cov_method (str, optional): [description]. Defaults to 'standard'.

        Returns:
            [type]: [description]
        """
        super().fit()
        sigma, cov, se = self._variance.fd(self._exog, self._resid,
                                          cov_method)
        t_values = self._b_hat / se
        names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
        values = [self._b_hat, se, sigma, t_values, self._R2, cov]
        self.results = dict(zip(names, values)) 
        return self.results

