import numpy as np
from numpy import linalg as la
from tabulate import tabulate
import pandas as pd
import re
import scipy.stats as stats

class Plm:
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
        models = ('pools', 'fd', 'fe', 're', 'be')
        assert model in models, \
            f'{model} is not a valid model name. Must be in {models}'
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
        self._model = model
        self._N, self._k = self._exog.shape
        self._t = t
    
    def fit(self):
        """
        Fits the model
        """
        self._b_hat = self._estimate()
        self._resid = self._resid()
        self._SSR = self._SSR()
        self._SST = self._SST()
        self._R2 = self._R2()
        self._adj_R2 = self._adjusted_R2()
        sigma2, cov, se = self._variance.var(self._SSR)
        if self.cov_method == 'robust':
            cov, se = self._variance.robust_var(self._resid)

        t_values = self._b_hat / se
        p_values = stats.t.sf(np.abs(t_values), self._N-1)*2
        names = ['b_hat', 'se', 'sigma2', 't_values', 'p_values', 'R2', 'adj_R2','cov']
        values = [self._b_hat, se, sigma2, t_values, p_values,self._R2, self._adj_R2, cov]
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
    def _adjusted_R2(self) ->float:
        return 1.0 - ((1-self._R2)*(self._N-1) / (self._N - self._k - 1))
    
    def _get_lam(self, dependent, exog, t, cov_method):
        sigma2_w = Plm(dependent, exog, 'be', t, cov_method).fit().results['sigma2']
        sigma2_u = Plm(dependent, exog, 'fe', t, cov_method).fit().results['sigma2']
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
        assert self.results
        if self.labels:
             label_y, label_x = self.labels
        elif labels is not None:
            label_y, label_x = labels
        
        assert isinstance(label_x, list), f'label_x must be a list (second part of the tuple, labels)'
        assert len(label_x) == self.results['b_hat'].size, \
        f'Number of labels for x should be the same as number of estimated parameters'
        
        # Create table, using the label for x to get a variable's coefficient,
        # standard error and t_value.
        def p_stars(pval):
            if pval < 0.01:
                return "***"
            elif pval < 0.05:
                return "**"
            elif pval < 0.1:
                return "*"
            else:
                return " "
        
        table = []
        for i, name in enumerate(label_x):
            row = [
                name, 
                str(round(self.results.get('b_hat')[i][0],3)) \
                +p_stars(self.results.get('p_values')[i]), 
                self.results.get('se')[i], 
                self.results.get('t_values')[i]
            ]
            table.append(row)
        
        tab = tabulate(table, headers, **kwargs, 
                       floatfmt=decimals)
        tlen = len(tab.split('\n')[0])    
        print(f"{' '*(int(tlen/2)-int(len(title)/2)-2)} {title}")
        print('_'*tlen)
        print(chr(8254)*tlen)
        print(f"Dependent variable: {label_y}\n")
        #print('_'*tlen)
        print(tab)
        print('_'*tlen, '\n')
        # Print extra statistics of the model.
        print(f"R\u00b2 = {self.results.get('R2').item():.3f}")
        print(f"Adj R\u00b2 = {self.results.get('adj_R2').item():.3f}")
        print(f"\u03C3\u00b2 = {self.results.get('sigma2').item():.3f}")
        
        models = {'fe':'Fixed effects', 'pools':'Pooled OLS',
                  'fd':'First-difference', 'be':'Between estimator',
                  're':'Random effects'}
        print(f"Model: {models[self._model]}")
        print(f"No. observations: {self._N}")
        print(f"No. timeperiods: {self._t}")
        if _lambda: 
            print(f'\u03bb = {_lambda.item():.3f}')
        print('_'*tlen)
        print(chr(8254)*tlen)
        print('Note: ∗p<0.1;∗∗p<0.05;∗∗∗p<0.01')
        if self.cov_method == 'robust':
            print(f'Heteroscedastic robust standard errors.')
        

class PlmFormula(Plm):
    
    def __init__(self, formula:str, model:str, cov_method:str='',
                 include_intercept:bool=False, data:pd.DataFrame=None):
        """Fit Plm using formula notation.
        Args:
            formula (str): example "dependent ~ exog_1 + exog_2. Also understands
            interactions and polynomials e.g. exog_1*exog_2 or exog_1^2.
            data (pd.DataFrame): dataframe containing the data for model.
            Must have multiindex of (1) obsid, (2) time.
        """
        dependent, exog, t, self.labels = self._parse_formula(formula, data, include_intercept)
        self._t, self._nobs = t, exog.shape[0]
        if model == 're':
            lam = self._get_lam(dependent, exog, t, cov_method)
        else:
            lam = None
        self.cov_method = cov_method
        self._dependent = Transform(dependent, t, model, lam).perm()
        self._exog = Transform(exog, t, model, lam).perm()
        self._variance = VarianceEstimator(self._exog, t, model)
        self._model = model
        self._N, self._k = self._exog.shape
        self._t = t
        
    def _parse_formula(self, formula, data, include_intercept):
        """Parse formula"""
        
        #TODO: assert correct format assert re.match(r'')
        assert type(data) == pd.core.frame.DataFrame, \
        'Data must Pandas dataframe object'
        assert type(data.index) == pd.core.indexes.multi.MultiIndex, \
        'Data must contain multi-index'
        N = len(np.unique([i[0] for i in data.index]))
        t = len(np.unique([i[1] for i in data.index]))
        assert data.shape[0] == N*t, 'Data is not a balanced panel'
        
        y, X = formula.replace(' ','').split('~')
        X = X.split('+')
        for v in X:
            assert v in data.columns, \
                f'Variable {v} does not exist in dataframe.'
            if '*' in v:
                v1,v2 = v.split('*')
                data[v] = data[v1] * data[v2]
            elif '^' in v:
                vp, grade = v.split('^')
                data[v] = data[vp]**int(grade)
        dependent = data[y].values.reshape(-1,1)
        exog = data[X].values
        
        if include_intercept:
            const = np.ones((N*t,1))
            exog = np.column_stack((const, exog))
            X.insert(0, 'intercept')
        
        labels = (y, X)
           
        return dependent, exog, t, labels
        
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