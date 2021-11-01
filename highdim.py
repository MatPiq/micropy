import numpy as np
import numpy.linalg as la 
import scipy as sp
from scipy.stats import norm
from sklearn.linear_model import Lasso, LinearRegression
from statsmodels.api import OLS
import pandas as pd




class LassoIV(object):
    """
    Implementation of the post-double lasso (ppl) and
    poost-partialing out lasso (ppol). Also includes lambda estimation
    according to the Bickel-Ritov-Tsybakov (brt) and
    Bellon-Chen-Chernozhukov-Hansen (bcch) rules.    
    
    Args:
        FILL IN
    Returns:
        FILL IN 
    """
    def __init__(self, y:str, d:str, data:pd.DataFrame,
                 estimator:str='ppol', lamb_method:str='brt',
                 normalize:bool=True, c:float=1.1):
        
        assert estimator in ['pdl', 'ppol'], f'Incorrect lambda method {lamb_method}'
        assert lamb_method in ['brt', 'bcch'], f'Incorrect lambda method {lamb_method}'
        assert type(data) == pd.core.frame.DataFrame
        self.data = data
        self.N, self.P = self.data.drop([y], axis=1).shape
        self.c = c
        self.estimator = estimator
        self.lamb_method = lamb_method
        self.y = data[y]
        self.d = data[d]
        self.Z = self._normalize(data.drop([y,d], axis = 1))
        print(f'For {y} ~ Z')
        self.lam_yZ = self._calc_lambda(self.y, self.Z)
        print(f'For {d} ~ Z')
        self.lam_dZ = self._calc_lambda(self.d, self.Z)
   
        if estimator == 'pdl':
            self.X = self._normalize(data.drop([y], axis = 1))
            print(f'For {y} ~ X')
            self.lam_yX = self._calc_lambda(self.y, self.X)
               
    def fit(self, significant_lvl:float=0.05, include_intercept:bool=True):
        
        lasso_yZ = Lasso(alpha=self.lam_yZ).fit(self.Z, self.y)
        lasso_dZ = Lasso(alpha=self.lam_dZ).fit(self.Z, self.d)
        resid_yZ = self.y - lasso_yZ.predict(self.Z)
        resid_dZ = self.d - lasso_dZ.predict(self.Z)
        resid_dZ2 = resid_dZ**2
        
        if self.estimator == 'ppol':
                     
            resid_yZdZ = resid_yZ*resid_dZ
           
            self.alpha=np.sum(resid_yZdZ)/np.sum(resid_dZ2) # partialling-out Lasso estimate
            self.sigma2 = (self.N*np.sum(resid_yZdZ ** 2)) / (np.sum(resid_dZ2) ** 2)
        
        elif self.estimator == 'pdl':
            lasso_yX = Lasso(alpha=self.lam_yX).fit(self.X, self.y)
            resid_yX = lasso_yX.predict(self.X)
            d_coef = lasso_yX.coef_[self.X.columns.get_loc(self.d.name)]
            resid_yZ_nod = self.y-resid_yX+(d_coef*self.d)
            reisd_dZyZ_nod = resid_yZ_nod*resid_dZ
            self.alpha = np.sum(reisd_dZyZ_nod) / np.sum(resid_dZ*self.d)
            resid_yXdZ = resid_yX * resid_dZ
            self.sigma2 = (self.N*np.sum(resid_yXdZ ** 2)) / (np.sum(resid_dZ2)**2) #Post-double estimate
            
        print('-'*66)
        print(f'{self.d.name} estimate: {np.round(self.alpha,6)}')     
        
        self.se = np.sqrt(self.sigma2 / self.N)
        quant = self.se*norm.ppf(np.abs(1-significant_lvl/2))
        self.ci = np.array([self.alpha-quant, self.alpha+quant])
        print(f'Confident interval: {np.round(self.ci,6)}')
        return self
    
    def _calc_lambda(self, y, X):
        P = X.shape[1]
        alpha = 0.05
        sigmahat = np.std(y)
        
        if self.lamb_method == 'brt':
            lam = self.c*sigmahat*norm.ppf(1-alpha/(2*P))/np.sqrt(self.N)
            print(f'The Bickel-Ritov-Tsybakov lambda is: {np.round(lam,5)}')
        
        elif self.lamb_method == 'bcch': 
            yX = (np.max((X.T ** 2) @ ((y-np.mean(y)) ** 2) / self.N)) ** 0.5
            lambda_pilot = self.c*norm.ppf(1-alpha/(2*P))*yX/np.sqrt(self.N)
            # Pilot estimates
            coef_pilot = Lasso(alpha=lambda_pilot).fit(X,y).coef_
            # Updated penalty
            res = y - X @ coef_pilot
            resXscale = (np.max((X.T ** 2) @ (res ** 2) / self.N)) ** 0.5
            lam = self.c*norm.ppf(1-alpha/(2*P))*resXscale/np.sqrt(self.N)
            print(f'The Belloni-Chen-Chernozhukov-Hansen lambda is: {np.round(lam,5)}')
            
        return lam
    
    @staticmethod
    def _normalize(M):
        return (M - M.mean(axis = 0)) / M.std(axis = 0)