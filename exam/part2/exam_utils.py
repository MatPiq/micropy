import numpy as np
from tqdm.notebook import tqdm
import scipy.stats as st
import time

def wald_test(beta, cov, R, r):
    """
    Calculates the Wald test statistic for a given beta and covariance matrix.
    args:
        beta(np.array): the beta vector
        cov(np.array): the covariance matrix
        R(float): the hypothesis test statistic
    returns:
        wald_stat(float): the Wald test statistic
        p_value(float): the p-value of the test
    """
    Q = R.shape[0]
    W = (R@beta-r).T@np.linalg.inv(R@cov@R.T)@(R@beta-r)
    p_vals = 1-st.chi2(Q).cdf(W)
    return W, p_vals

def calc_r2(y,ypred):
    """
    Calculates R-squared for a given set of observations and predictions.
    args:
        y: observed values
        ypred: predicted values
    returns:
        r2: R-squared
    """
    res = np.sum((y-ypred)**2)
    tot = np.sum((y-y.mean())**2)
    return 1-(res/tot)

def partial_effects(model, y,x, deriv, x0:np.array, 
                    N_bootstraps:int=500)->np.array:
    """
    Calculates the average partial effects of betas evaluated
    at x⁰.
    args:
        model(Mestimator):
        y(np.array): the outcome variable
        x(np.array): the regressors
        deriv(func): the derivative of the model w.r.t. x's
        x0(np.array): the x values to be evaluated
    returns:
        partial_effects(np.array): the partial effects of size K regressors N obs in x0
        se_partial_effect(np.array): the se of partial effects of size K regressors N obs in x0
    """
    def calc_partials(model, y,x, deriv, x0:np.array):
        fitted = model.fit(y,x, cov_type='None', options = {'disp': False})
        betas = fitted.res['theta'][:x.shape[1]]
        partials = deriv(x0,betas)
        return partials
    
    partials = calc_partials(model, y,x, deriv, x0)
    se_partials = np.empty((N_bootstraps, x0.shape[0], x.shape[1]))
    
    np.random.seed(42)
    for i in tqdm(range(N_bootstraps)):
        sample_idx = np.random.choice(y.shape[0], 
                                      size=y.shape[0], 
                                      replace=True)
        
        x_sample = x[sample_idx]
        y_sample = y[sample_idx]
        se_partials[i] = calc_partials(model, y_sample, x_sample, deriv, x0)
    
    return partials, se_partials.std(axis=0)
    