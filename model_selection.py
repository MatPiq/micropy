#from linearmodels import Transform, OLS
import numpy as np
import numpy.linalg as la
from scipy.stats import chi2
from tabulate import tabulate



def serial_correlation(e,t:int):
    """Evaluates serial correlation between...
    """

    
    L_T = np.eye(t, k=-1)
    L_T = L_T[1:]
    e_l = perm(L_T, e)
    e = np.delete(e, list(range(0, e.shape[0], t)), axis=0)
    #e = np.delete(e, list(range(0, e.shape[0], t-1)), axis=0)
    return e, e_l
    return estimate(e, e_l)
    
def hausman_test(fe, re, print_summary=False):
    
    fe_betas = fe['b_hat']
    re_betas =  re['b_hat'][fe_betas.shape[0]:]
    #assert fe_betas.shape[0] == re_betas.shape[0], \
    #f'Dim {fe_betas.shape[0]} and {re_betas.shape[0]} must match!'
    fe_cov =  fe['cov']
    re_cov = re['cov'][fe_cov.shape[0]:,fe_cov.shape[1]:]
    #return re_cov, fe_cov, re_betas, fe_betas
    diff = fe_betas-re_betas
    H = diff.T@la.inv(fe_cov-re_cov)@diff
    p_val = chi2.sf(H.item(), 4)
    
    if print_summary:
        table = []
        for i in range(len(diff)):
            row = [
                fe_betas[i], re_betas[i], diff[i]
            ]
            table.append(row)

        print(tabulate(
            table, headers=['b_fe', 'b_re', 'b_diff'], floatfmt='.4f'
            ))
        print(f'The Hausman test statistic is: {H.item():.2f}, with p-value: {p_val:.2f}.')
    
    
    return H, p_val, diff
    

def perm(Q_T, A:np.array, t=0) -> np.array:
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
        
        if t == 0:
            t = Q_T.shape[1]

        # Loop over the individuals, and permutate their values.
        for i in range(int(A.shape[0]/t)):
            Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
        return Z
    
def estimate( 
        y: np.array, x: np.array, transform='', t:int=None
    ) -> list:
    """Uses the provided estimator (mostly OLS for now, and therefore we do 
    not need to provide the estimator) to perform a regression of y on x, 
    and provides all other necessary statistics such as standard errors, 
    t-values etc.  

    Args:
        >> y (np.array): Dependent variable (Needs to have shape 2D shape)
        >> x (np.array): Independent variable (Needs to have shape 2D shape)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >>t (int, optional): If panel data, t is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov'
    """
    
    b_hat = est_ols(y, x)  # Estimated coefficients
    residual = y - x@b_hat  # Calculated residuals
    SSR = residual.T@residual  # Sum of squared residuals
    SST = (y - np.mean(y)).T@(y - np.mean(y))  # Total sum of squares
    R2 = 1 - SSR/SST

    sigma2, cov, se = variance(transform, SSR, x, t)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma2, t_values, R2, cov]
    return dict(zip(names, results))

def est_ols( y: np.array, x: np.array) -> np.array:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.array): Dependent variable (Needs to have shape 2D shape)
        >> x (np.array): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.array, 
        t: int
    ) -> tuple:
    """Calculates the covariance and standard errors from the OLS
    estimation.

    Args:
        >> transform (str): Defaults to ''. If the data is transformed in 
        any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation
        >> SSR (float): Sum of squared residuals
        >> x (np.array): Dependent variables from regression
        >> t (int): The number of time periods in x.

    Raises:
        Exception: If invalid transformation is provided, returns
        an error.

    Returns:
        tuple: Returns the error variance (mean square error), 
        covariance matrix and standard errors.
    """

    # Store n and k, used for DF adjustments.
    k = x.shape[1]
    if transform in ('', 'fd', 'be'):
        n = x.shape[0]
    else:
        n = x.shape[0]/t

    # Calculate sigma2
    if transform in ('', 'fd', 'be'):
        sigma2 = (np.array(SSR/(n - k)))
    elif transform.lower() == 'fe':
        sigma2 = np.array(SSR/(n * (t - 1) - k))
    elif transform.lower() == 're':
        sigma2 = np.array(SSR/(t * n - k))
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma2, cov, se