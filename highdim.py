import numpy as np
import numpy.linalg as la 
import scipy as sp
class LinReg:
    def __init__(self, estimation = 'ols', lam:float=0.05, 
                 fit_intercept=True, optimizer='BFGS'):
        
        assert estimation in ('ols', 'lasso', 'ridge', 'elasticnet'), \
        f'Incorrect model {estimation}'
        self.optim = optimizer
        self.estimation = estimation
        self.fit_intercept = fit_intercept
        self.lam = lam
        
    def fit(self, X, y):
        if self.estimation == 'ols':
            return la.inv(X.T@X)@(X.T@y)
        else:
            w = np.random.rand(X.shape[1],1)
            #obj_fnc = self._objective(X,y,w)
            res = sp.optimize.minimize(self._objective, w,
                            args=(X,y,self.lam),
                            method='BFGS')#
            return res
            
    
    def predict(self, X_new):
        pass
    
    def _objective(self, w, X:np.array, y:np.array, lam):
        """
        lasso: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        """
        if self.estimation == 'lasso':
            return np.linalg.norm(y - X@w)**2 + self.lam * np.sum(abs(w))