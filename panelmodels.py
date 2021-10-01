import numpy as np
from numpy import linalg as la
from numpy.lib.function_base import cov
from tabulate import tabulate

class PLM:
   
    def __init__(self, dependent:np.array, exog:np.array, model:str = 'pools',
                 t:int = 1, cov_method=''):
        """Linear Panel models

        Args:
            dependent (np.array): N*1 vector of the dependent variable
            exog (np.array): N*K matrix of exogenous variables
            model (str, optional): Which model to choose. Defaults to 'pools'.
            other options:
                - ("fe"): Fixed Effects estimator
                - ("fd"): First Difference estimator
                - ("re"): Random Effects estimator
                - ("be"): Between estimator
            t (int, optional): Amount of timeperiods in data. Defaults to 1.
            cov_method (str, optional): Type of covariance matrix to be calculated. Defaults to ''.
        """
        assert dependent.ndim == 2, 'Input y must be 2-dimensional'
        assert exog.ndim == 2, 'Input x must be 2-dimensional'
        assert dependent.shape[1] == 1, 'y must be a column vector'
        assert dependent.shape[0] == exog.shape[0], \
        'y and x must have same first dimension'
        
        if model == 're':
            lam = self._get_lam(dependent, exog, t, cov_method)
        else:
            lam = None
        self.cov_method = cov_method
        self._dependent = Transform(dependent, t, model, lam).perm()
        self._exog = Transform(exog, t, model, lam).perm()
        self._variance = VarianceEstimator(self._exog, t, model)
    
    def fit(self):
        """
        Fits the model
        """
        self._b_hat = self._estimate()
        self._resid = self._resid()
        self._SSR = self._SSR()
        self._SST = self._SST()
        self._R2 = self._R2()
        sigma2, cov, se = self._variance.var(self._SSR)
        if self.cov_method == 'robust':
            cov, se = self._variance.robust_var(self._resid)

        t_values = self._b_hat / se
        names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov']
        values = [self._b_hat, se, sigma2, t_values, self._R2, cov]
        self.results = dict(zip(names, values)) 
        return self
    
    def predict(self, X_new:np.array) -> np.array:
        """Uses estimated betas to predict y hat from an array X"""
        return X_new@self._b_hat
    
    def _estimate(self) -> np.array:
        """
        Estimates OLS for various models. Data can be transformed.
        Math: (X'X)^{-1}(X'y)
        """
        XX = la.inv(self._exog.T@self._exog)
        Xy = self._exog.T@self._dependent
        return XX@Xy
    
    def _resid(self) -> np.array:
        return self._dependent - self._exog@self._b_hat

    def _SSR(self) -> float:
        return np.sum(self._resid **2)

    def _SST(self)-> float:
        return np.sum((self._dependent-self._dependent.mean())**2)

    def _R2(self)-> float:
        return 1.0 - (self._SSR / self._SST)
    
    #@classmethod
    def _get_lam(self, dependent, exog, t, cov_method):
        sigma2_w = PLM(dependent, exog, 'be', t, cov_method).fit().results['sigma2']
        sigma2_u = PLM(dependent, exog, 'fe', t, cov_method).fit().results['sigma2']
        return 1 - np.sqrt(sigma2_u / (sigma2_u + (t-1)*sigma2_w))
    
    def summary(self,
        labels: tuple=None,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        decimals='.3f',
        _lambda:float=None,
        **kwargs
        ) -> None:
        """Prints a nice looking table, must at least have coefficients, 
        standard errors and t-values. The number of coefficients must be the
        same length as the labels.

        Args:
            >> labels (tuple): Touple with first a label for y, and then a list of 
            labels for x.
            >> results (dict): The results from a regression. Needs to be in a 
            dictionary with at least the following keys:
                'b_hat', 'se', 't_values', 'R2', 'sigma2'
            >> headers (list, optional): Column headers. Defaults to 
            ["", "Beta", "Se", "t-values"].
            >> title (str, optional): Table title. Defaults to "Results".
            _lambda (float, optional): Only used with Random effects. 
            Defaults to None.
        """
        if not self.results:
            raise Exception(f'Model has not been fitted on data yet!')
        
        # Unpack the labels
        label_y, label_x = labels
        assert isinstance(label_x, list), f'label_x must be a list (second part of the tuple, labels)'
        assert len(label_x) == self.results['b_hat'].size, \
        f'Number of labels for x should be the same as number of estimated parameters'
        
        # Create table, using the label for x to get a variable's coefficient,
        # standard error and t_value.
        table = []
        for i, name in enumerate(label_x):
            row = [
                name, 
                self.results.get('b_hat')[i], 
                self.results.get('se')[i], 
                self.results.get('t_values')[i]
            ]
            table.append(row)
        
        # Print the table
        print(title)
        print(f"Dependent variable: {label_y}\n")
        print(tabulate(table, headers, **kwargs, floatfmt=decimals))
        
        # Print extra statistics of the model.
        print(f"R\u00b2 = {self.results.get('R2').item():.3f}")
        print(f"\u03C3\u00b2 = {self.results.get('sigma2').item():.3f}")
        if _lambda: 
            print(f'\u03bb = {_lambda.item():.3f}')

class VarianceEstimator:
    def __init__(self, exog:np.array, t:int, model:str = ''):
        """Calculates the variance"""
        self.exog = exog
        self.t = t
        self.model = model
        self.nrows, self.k = exog.shape
    
    def corr_denom(self) -> int:
        """Corrects the denominator based on transformation"""
        if self.model in ('', 'pools', 'fd', 'be', 're'):
            denom = self.nrows - self.k 
        elif self.model == 'fe': 
            n = self.nrows/self.t
            denom = self.nrows - n - self.k 
        return denom
      
    def var(self, SSR:int) -> tuple:
        """Calculates the normal variance estimator"""
        sigma2 = SSR / self.corr_denom()
        cov = sigma2*la.inv(self.exog.T@self.exog)
        se = np.sqrt(cov.diagonal()).reshape(-1, 1)
        return sigma2, cov, se
    
    def robust_var(self,residual: np.array) -> tuple:
        """Calculates the robust variance estimator"""
        
        # If only cross sectional, we can easily use the diagonal.
        if not self.t:
            Ainv = la.inv(self.exog.T@self.exog)
            uhat2 = residual ** 2
            uhat2_x = uhat2 * self.exog # elementwise multiplication: avoids forming the diagonal matrix (RAM intensive!)
            cov = Ainv @ (self.exog.T@uhat2_x) @ Ainv
        # Else we loop over each individual.
        else:
            n = int(self.nrows / self.t)        
            B = np.zeros((self.k, self.k)) 
            # initialize         
            for i in range(n):            
                idx_i = slice(i*self.t, (i+1)*self.t) # index values for individual i             
                Omega = residual[idx_i]@residual[idx_i].T            
                B += self.exog[idx_i].T @ Omega @ self.exog[idx_i]        
                Ainv = la.inv(self.exog.T @ self.exog)        
                cov = Ainv @ B @ Ainv
        
        se = np.sqrt(np.diag(cov)).reshape(-1, 1)
        return cov, se

class Transform:
    
    def __init__(self, A:np.array, t:int, model:str,
                 lam:int):
        self.A = A
        self.t = t
        self.model = model
        self.lam = lam
     
    def get_Q_T(self):
        if self.model == 'fe':
            return np.identity(self.t) - np.tile(1/self.t, (self.t, self.t))
        elif self.model == 'fd':
            return (np.eye(self.t) - np.eye(self.t, k=-1))[1:]
        elif self.model == 'be':
            return np.tile(1/self.t, (self.t, self.t))
        elif self.model == 're':
            return np.eye(self.t) - self.lam*np.ones(self.t)/self.t
    
    def perm(self) -> np.array:
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
        # We can infer t from the shape of the transformation matrix.
        if self.model == 'pools':
            return self.A
        else:
            Q_T = self.get_Q_T()
            M,T = Q_T.shape 
            N = int(self.A.shape[0]/T)
            K = self.A.shape[1]
            # initialize output 
            Z = np.empty((M*N, K))
            
            for i in range(N): 
                ii_A = slice(i*T, (i+1)*T)
                ii_Z = slice(i*M, (i+1)*M)
                Z[ii_Z, :] = Q_T @ self.A[ii_A, :]

            return Z