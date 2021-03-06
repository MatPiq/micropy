#from linearmodels import Transform, OLS
import numpy as np
import numpy.linalg as la
from scipy.stats import chi2
from tabulate import tabulate
import LinearModelsWeek3_post as lm
from panel import Plm

def wald_test(model):
    b_hat = model.results['b_hat']
    cov = model.results['cov']
    R = np.ones(b_hat.shape[0]).reshape(1,-1)
    W = (R@b_hat-1).T@la.inv(R@cov@R.T)@(R@b_hat-1)
    print(f'Wald test statistic: {W[0][0]:.5f}')
    print(f'P-value on Wald test: {chi2(1).pdf(W)[0][0]:.5f}')

def serial_correlation(y,X,t:int):
    """
    Evaluates if serial correlation exists.
    """
    t = t-1
    b_hat = lm.est_ols(y, X)
    e = y - X@b_hat
    L_T = np.eye(t, k=-1)
    L_T = L_T[1:]
    e_l = lm.perm(L_T, e)
    e = np.delete(e, list(range(0, e.shape[0], t)), axis=0)
    label_ye = 'OLS residual, e\u1d62\u209c'
    label_e = ['e\u1d62\u209c\u208B\u2081']
    Plm(dependent=e, exog=e_l, model='pools').fit().summary(labels=(label_ye, label_e))
    
    
def hausman_test(fe, re, print_summary=False, headers = ['b_fe', 'b_fd']):
    
    #Check if re include time invariants
    shape_diff = len(re.results['b_hat']) - len(fe.results['b_hat'])
    if shape_diff != 0:
        re_betas = re.results['b_hat'][shape_diff:]
        re_cov = re.results['cov'][shape_diff:, shape_diff:]
    else:
        re_betas =  re.results['b_hat']
        re_cov = re.results['cov']
        
    fe_betas = fe.results['b_hat']
    fe_cov =  fe.results['cov']
    hat_diff = fe_betas-re_betas
    cov_diff = la.inv(fe_cov-re_cov)
    H = hat_diff.T@cov_diff@hat_diff
    p_val = chi2.sf(H.item(), 4)
    
    if print_summary:
        table = []
        for i in range(len(hat_diff)):
            row = [
                fe_betas[i], re_betas[i], hat_diff[i]
            ]
            table.append(row)

        print(tabulate(
            table, headers= headers+['b_diff'], floatfmt='.4f'
            ))
        print(f'The Hausman test statistic is: {H.item():.2f}, with p-value: {p_val:.2f}.')
