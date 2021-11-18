import numpy as np
import numpy.linalg as la 
import scipy as sp
from scipy.stats import norm
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()


class LassoIV(object):
    """
    Implementation of the post-double lasso (ppl) and
    poost-partialing out lasso (ppol). Also includes lambda estimation
    according to the Bickel-Ritov-Tsybakov (brt) and
    Bellon-Chen-Chernozhukov-Hansen (bcch) rules.    
    
    Args:
        y(str): dependent variable
        d(str): treatment variable
        data(pd.DataFrame): dataframe with y,d and controls
        estimator(str): 'ppol' or 'pdl'
        lamb_method(str): 'brt' or 'bcch'
        normalize(bool): if True, normalize Z
        c(float): tuning parameter
        max_iter(int): maximum number of iterations
    """
    def __init__(self, y:str, d:str, data:pd.DataFrame,
                 estimator:str='ppol', lamb_method:str='brt',
                 normalize:bool=True, c:float=1.1, max_iter=1000):
        
        assert estimator in ['pdl', 'ppol'], f'Incorrect lambda method {lamb_method}'
        assert lamb_method in ['brt', 'bcch'], f'Incorrect lambda method {lamb_method}'
        assert type(data) == pd.core.frame.DataFrame
        self.data = data
        self.N, self.P = self.data.drop([y], axis=1).shape
        self.c = c
        self.estimator = estimator
        self.lamb_method = lamb_method
        self.max_iter = max_iter
        self.y = data[y]
        self.d = data[d]
        self.Z = data.drop([y,d], axis = 1)
        self.Z = pd.DataFrame(scaler.fit_transform(self.Z), 
                              index=self.Z.index, columns=self.Z.columns)
        #if normalize:
        #    self.Z = self._normalize(self.Z)
        print(f'For {y} ~ Z')
        self.lam_yZ = self._calc_lambda(self.y, self.Z)
        print(f'For {d} ~ Z')
        self.lam_dZ = self._calc_lambda(self.d, self.Z)
   
        if estimator == 'pdl':
            self.X = data.drop([y], axis = 1)
            
            self.X = pd.DataFrame(scaler.fit_transform(self.X),
                                  index=self.X.index, columns=self.X.columns)
           
            print(f'For {y} ~ X')
            self.lam_yX = self._calc_lambda(self.y, self.X)
               
    def fit(self, significant_lvl:float=0.05):
        """
        Fits the model.
        Args:
            significant_lvl (float, optional): Defaults to 0.05.
        """
        
        lasso_yZ = Lasso(alpha=self.lam_yZ, max_iter=self.max_iter).fit(self.Z, self.y)
        lasso_dZ = Lasso(alpha=self.lam_dZ, max_iter=self.max_iter).fit(self.Z, self.d)
        resid_yZ = self.y - lasso_yZ.predict(self.Z)
        resid_dZ = self.d - lasso_dZ.predict(self.Z)
        resid_dZ2 = resid_dZ**2
        
        self.yZ_coef = lasso_yZ.coef_
        self.dZ_coef = lasso_dZ.coef_
        print(f'Number of non-zero controls y ~ Z: {np.count_nonzero(lasso_yZ.coef_)}')
        print(f'Number of non-zero controls d ~ Z: {np.count_nonzero(lasso_dZ.coef_)}')
        
        if self.estimator == 'ppol':    
            resid_yZdZ = resid_yZ*resid_dZ
            self.alpha=np.sum(resid_yZdZ)/np.sum(resid_dZ2) # partialling-out Lasso estimate
            self.sigma2 = (self.N*np.sum(resid_yZdZ ** 2)) / (np.sum(resid_dZ2) ** 2)
        
        if self.estimator == 'pdl':
            lasso_yX = Lasso(alpha=self.lam_yX, max_iter=self.max_iter).fit(self.X, self.y)
            print(f'Number of non-zero controls y ~ X: {np.count_nonzero(lasso_yX.coef_)}')
            self.yX_coef = lasso_yX.coef_
            pred_yX = lasso_yX.predict(self.X)
            d_coef = lasso_yX.coef_[self.X.columns.get_loc(self.d.name)]
            resid_yZ_nod = self.y-pred_yX+(d_coef*self._normalize(self.d))
            reisd_dZyZ_nod = resid_yZ_nod*resid_dZ
            self.alpha = np.sum(reisd_dZyZ_nod) / np.sum(resid_dZ*self.d)
            resid_yXdZ = (self.y - pred_yX) * resid_dZ
            self.sigma2 = self.N*np.sum(resid_yXdZ ** 2) / (np.sum(resid_dZ2)**2) #Post-double estimate

        self.se = np.sqrt(self.sigma2 / self.N)
        quant = self.se*norm.ppf(np.abs(1-(significant_lvl/2)))
        self.ci = np.array([self.alpha-quant, self.alpha+quant])
        print(f'{self.d.name} estimate: {np.round(self.alpha, 5)}')
        print(f'Confidence interval: {np.round(self.ci,5)}')
        return self
    
    def _calc_lambda(self, y, X):
        P = X.shape[1]
        alpha = 0.05
        
        if self.lamb_method == 'bcch': 
            yX = (np.max((X.T ** 2) @ ((y-np.mean(y)) ** 2) / self.N)) ** 0.5
            lambda_pilot = self.c*norm.ppf(1-alpha/(2*P))*yX/np.sqrt(self.N)
            # Pilot estimates
            coef_pilot = Lasso(alpha=lambda_pilot, max_iter=self.max_iter).fit(X,y).coef_
            # Updated penalty
            res = y - X @ coef_pilot
            resXscale = (np.max((X.T ** 2) @ (res ** 2) / self.N)) ** 0.5
            lam = self.c*(norm.ppf(1-alpha/(2*P))*resXscale/np.sqrt(self.N))
            print(f'The Belloni-Chen-Chernozhukov-Hansen lambda is: {np.round(lam,5)}')
        
        if self.lamb_method == 'brt':
            sigmahat = np.std(y)
            lam = self.c*sigmahat*norm.ppf(1-alpha/(2*P))/np.sqrt(self.N)
            print(f'The Bickel-Ritov-Tsybakov lambda is: {np.round(lam,5)}')
        
     
            
        return lam
    
    @staticmethod
    def _normalize(M):
        return (M - M.mean(axis = 0)) / M.std(axis = 0)