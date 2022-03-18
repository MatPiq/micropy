import numpy as np
from numpy.core.fromnumeric import size
import scipy.stats as st
from numpy import linalg as la
import pandas as pd 
from scipy import optimize
from tabulate import tabulate
import abc
import time 


class MestimatorBase(abc.ABC):
    
    @abc.abstractmethod
    def starting_values(self, y, X) -> np.array:
        """
        Returns a "reasonable" vector of parameters from which to start estimation
        Returns:
            theta0: (K,) vector of starting values for estimation
        """

    @abc.abstractmethod
    def loglikelihood(self, theta, y, X) -> np.array:
        """
        Computes the log-likelihood contribution.
        """

    def criterion(self, theta,  y, X) -> np.array:
        """
        Computes the criterion function (negative loglikelihood)
        """
        return - self.loglikelihood(theta, y, X)

    @abc.abstractmethod
    def sim_data(self, theta:np.array, N:int) -> tuple:
        """
        Simulate from the assumed data generating process.
        """
    def fit(self,
        y: np.ndarray, 
        X: np.ndarray, 
        cov_type='outer product',
        options = {'disp': True},
        **kwargs
        ) -> dict:
        """Takes a function and returns the minimum, given start values and 
        variables to calculate the residuals.

        Args:
            q: The function to minimize. Must return an (N,) vector.
            theta0 (list): A list with starting values.
            y (np.array): Array of dependent variable.
            x (np.array): Array of independent variables.
            cov_type (str, optional): String for which type of variances to 
            calculate. Defaults to 'Outer Product'.
            options: dictionary with options for the optimizer (e.g. disp=True,
            which tells it to display information at termination.)

        Returns:
            dict: Returns a dictionary with results from the estimation.
        """
    
        # The minimizer can't handle 2-D arrays, so we flatten them to 1-D.
        theta0 = self.starting_values(y,X).flatten()
        P = theta0.size
        N = y.size

        # The objective function is the average of q(), 
        # but Q is only a function of one variable, theta, 
        # which is what minimize() will expect
        Q = lambda theta: np.mean(self.criterion(theta, y, X))

        # call optimizer
        result = optimize.minimize(Q, theta0, options=options, **kwargs)
        
        # collect output in a dict 
        res = {
            'theta':    result.x,
            'success':  result.success, # bool, whether convergence was succesful 
            'nit':      result.nit, # no. algorithm iterations 
            'nfev':     result.nfev, # no. function evaluations 
            'fun':      result.fun # function value at termination 
        }

        try: 
            cov, se = self.variance(self.criterion, y, X, res['theta'], cov_type)      
        except Exception as e: 
            if cov_type != 'None':
                print(f'Failed to compute std. errs.: got error "{e}"')
            cov = np.nan*np.ones((P,P))
            se = np.nan*np.ones((P,))

        res['se']   = se
        res['t']    = result.x / se
        res['cov']  = cov
        self.res = res
        return self

    def variance(self,
        q, # function taking three inputs: q(theta, y, x) 
        y: np.ndarray, 
        x: np.ndarray, 
        thetahat: np.ndarray, 
        cov_type: str
        ) -> tuple:
        """Calculates the variance for the likelihood function.

        Args:
            >> q: Function taking three inputs, q(theta,y,x), where we minimize the mean of q.
            >> y (np.ndarray): Dependent variable.
            >> x (np.ndarray): Independent variables.
            >> theta (hnp.ndarray): (K,) array
            >> cov_type (str): Type of calculation to use in estimation.

        Returns:
            tuple: Returns the variance-covariance matrix and standard errors.
        """
        N = y.size
        P = thetahat.size

        # numerical gradients 
        f_q = lambda theta : q(theta,y,x) # as q(), but only a function of theta, whereas q also takes y,x as inputs

        if cov_type in ['Outer Product', 'Sandwich']: 
            s = centered_grad(f_q, thetahat)
            B = (s.T@s)/N
        if cov_type in ['Hessian', 'Sandwich']: 
            H = hessian(f_q, thetahat)
            A = H/N
        
        # cov: P*P covariance matrix of theta 
        if cov_type == 'Hessian':
            cov = 1/N * np.linalg.inv(A)
        elif cov_type == 'Outer Product':
            cov = 1/N * np.linalg.inv(B)
        elif cov_type == 'Sandwich':
            # there are two ways of computing the Sandwich matrix product
            
            # method 1: simple to read, bad numerically
            # A_inv = np.linalg.inv(A)
            # cov = 1/N * (A_inv @ B @ A_inv)

            # metohd 2: hard to read, good numerically 
            Ainv_B = np.linalg.solve(A, B)
            Ainv_B_Ainv = np.linalg.solve(A.T, Ainv_B.T).T
            cov = 1/N * Ainv_B_Ainv


        # se: P-vector of std.errs. 
        se = np.sqrt(np.diag(cov))

        return cov, se
    
    def summary(self,
        theta_label: list,
        headers:list = ["", "Beta", "Se", "t-values"],
        title:str = "Results",
        num_decimals:int = 4
        ) -> None:
        """Prints a nice looking table, must at least have coefficients, 
        standard errors and t-values. The number of coefficients must be the
        same length as the labels.

        Args:
            theta_label (list): List of labels for estimated parameters
            results (dict): The output from estimate()
            dictionary with at least the following keys:
                'theta', 'se', 't'
            headers (list, optional): Column headers. Defaults to 
                ["", "Beta", "Se", "t-values"].
            title (str, optional): Table title. Defaults to "Results".
            num_decimals: (int) where to round off results (=None to disable)
        """
        assert self.res
        assert len(theta_label) == len(self.res['theta'])
        
        tab = pd.DataFrame({
            'theta': self.res['theta'], 
            'se': self.res['se'], 
            't': self.res['t']
            }, index=theta_label)
        
        if num_decimals is not None: 
            tab = tab.round(num_decimals)
        
        # Print the table
        opt_outcome = 'succeded' if self.res['success'] else 'failed'
        print(f'Optimizer {opt_outcome} after {self.res["nit"]} iter. ({self.res["nfev"]} func. evals.). Final criterion: {self.res["fun"]: 8.4g}.')
        print(title)
        return tab 

class Part1Estimator(MestimatorBase):

    def __init__(self, estimator:str='exp', estimate_sigma:bool=True,):
        assert estimator in ['exp', 'norm'],\
        f'Estimator must be one of exp, norm. Got {estimator}'
        self.estimator = estimator
        self.estimate_sigma = estimate_sigma

    def starting_values(self, y, X) -> np.array:

        return np.ones(X.shape[1]+1)#la.inv(X.T@X)@X.T@y
    
    def unpack(self,theta):
        beta = theta[:-1]
        sigma = theta[-1]
        return beta, sigma

    def loglikelihood(self, theta, y, X):
        beta, sigma = self.unpack(theta)
        if self.estimator == 'exp':
            Gxb = np.exp(-(X @ beta))
        
        if self.estimator == 'norm':
            Gxb = 3*st.norm.cdf(X @ beta)

        if self.estimate_sigma:
            t1 =np.log(sigma * np.sqrt(2.0*np.pi))
            t1 = np.fmax(t1, 1e-10) 
            ll = -t1 - ((y-Gxb)**2)/(2.0*sigma**2)
        else:
            ll = - ((y-Gxb)**2)
        return ll

    def predict(self, X_new):
        assert self.res
        beta = self.res['theta'][:-1]
        if self.estimator == 'norm':
            return 3*st.norm.cdf(X_new @ beta)
        if self.estimator == 'exp':
            return np.exp(-(X_new @ beta))

    def sim_data(self, theta, N):
        pass

class Part2Estimator(MestimatorBase):
    
    def __init__(self, estimator:str='exp', estimate_sigma:bool=True,
                seed=None, R=100):
        assert estimator in ['exp', 'norm'],\
        f'Estimator must be one of exp, norm. Got {estimator}'
        self.estimator = estimator
        self.seed = seed
        self.R = R
    
    def h(self,beta, x, c):
        if self.estimator == 'exp':
            res =  np.exp(-(x@beta+c))
        if self.estimator == 'norm':
            res = 3*st.norm.cdf(x@beta+c)
            
        if (res == 0).any():
            res = np.inf
        return res

    def criterion(self,theta, y, x):
        '''q(): criterion function to be minimized 
        '''
        return -self.loglikelihood(theta, y, x, self.seed, self.R)

    def loglikelihood(self, theta, y, x,  seed=None, R=100): 
        '''loglikelihood for the linear panel random effects model 
        Args
            theta: (K+2,) parameter vector with ordering: [beta, sigma_u, sigma_c]
            y: (N,T) matrix of outcomes 
            x: (N,T,K) matrix of covariates 
            seed: None or int: seed for np.random
                If None, equi-probable draws are used (a linspace over (0;1).)
            R: number of draws 
        
        Returns
            ll_i: (N,) vector of loglikelihood contributions
        '''
            
        assert theta.ndim == 1 
        assert x.ndim == 3
        
        N,T,K = x.shape

        # unpack params
        beta, sigma_e, sigma_c = self.unpack(theta)
        # draw c terms 
        c_draws = self.draw_c(sigma_c, N, R)
        f_ir = np.empty((N, R))
        for r in range(R):
            # draw c terms
            c_r = c_draws[:,r].reshape(N,1)
            # residual 
            res = (y-self.h(beta, x, c_r))**2
            # Sum residuals over T
            sum_res = np.sum(res, axis=1)
            # Multiply by rest of integral chunk
            f_ir[:,r] = np.exp(-1/(2*sigma_e**2)*sum_res)
        
        # mean over simulations 
        f_i = np.mean(f_ir, axis=1) # -> (N,)
        #Avoid taking log of 0
        f_i = np.fmax(f_i, 1e-10) 
        ll_i = (-T*np.log(sigma_e) - T/2*np.log(2*np.pi) + np.log(f_i))
        
        return ll_i

    def starting_values(self, y, X) -> np.array:

        return np.ones(X.shape[2]+2)

    def unpack(self,theta):
        beta = theta[:-2]
        sigma_e = np.abs(theta[-2])
        sigma_c = np.abs(theta[-1])
        return beta, sigma_e, sigma_c

    def draw_c(self,sigma_c, N, R, seed=None):
        '''draw_c: Draw the unobserved effect
        Args:
            seed: If None: R equi-probable draws are used 
                if int: np.random.normal is used 
                
        Returns: 
            c_draws: (N,R) matrix 
        '''
        if self.seed is not None:
            np.random.seed(seed)
            c_draws = sigma_c * np.random.normal(size=(N,R))
        else:
            uu = np.linspace(0,1,R+2)[1:-1] # 0 to 1, excluding end points 
            nn = sigma_c * st.norm.ppf(uu) # inverse normal CDF => normas
            c_draws = np.tile(nn, (N,1)) # -> result is (N,R)    return c
        
        return c_draws # (N,R)
   
    def predict(self, X_new):
        assert self.res
        beta = self.res['theta'][:-2]
        if self.estimator == 'norm':
            return 3*st.norm.cdf(X_new @ beta)
        if self.estimator == 'exp':
            return np.exp(-(X_new @ beta))

    def sim_data(self, theta, N, func, T=10):
        '''sim_data: simulate a dataset from the linear RE model 
        '''
        
        K = theta.size-2
        assert theta.ndim == 1 
        assert K>1, f'Got {K=}, expecting > 2'

        NT = int(N*T)

        # unpack params
        beta, sigma_u, sigma_c = self.unpack(theta)

        # sim x 
        oo = np.ones((NT,1))
        xx = np.random.normal(size=(NT,K-1))
        x = np.hstack([oo,xx]).reshape(N,T,K)

        # draw unobserved terms 
        c = sigma_c * np.random.normal(size=(N,1)) # crucial with (N,1): allows us to add it to (N,T) matrices correctly
        u = sigma_u * np.random.normal(size=(N,T))
        
        y = self.h(beta, x, 0, func)
    
        return y,x,c



def hessian(fhandle , x0 , h=1e-5 ) -> np.ndarray: 
    '''hessian(): computes the (K,K) matrix of 2nd partial derivatives
        using the aggregation "sum" (i.e. consider dividing by N)

    Args: 
        fhandle: callable function handle, returning an (N,) vector or scalar
            (i.e. you can q(theta) or Q(theta).)
        x0: K-array of parameters at which to evaluate the derivative 

    Returns: 
        hess: (K,K) matrix of second partial derivatives 
    
    Example: 
        from scipy.optimize import rosen, rosen_der, rosen_hess
        > x0 = np.array([-1., -4.])
        > rosen_hess(x0) - estimation.hessian(rosen, x0)
        The default step size of h=1e-5 gives the closest value 
        to the true Hessian for the Rosenbrock function at [-1, -4]. 
    '''

    # Computes the hessian of the input function at the point x0 
    assert x0.ndim == 1 , f'x0 must be 1–dimensional'
    assert callable(fhandle), 'fhandle must be a callable function handle'

    # aggregate rows with a raw sum (as opposed to the mean)
    agg_fun = np.sum

    # Initialization
    K = x0.size
    f2 = np.zeros((K,K)) # double step
    f1 = np.zeros((K,))  # single step
    h_rel = h # optimal step size is smaller than for gradient
                
    # Step size 
    dh = np.empty((K,))
    for k in range(K): 
        if x0[k] == 0.0: # use absolute step when relative is impossible 
            dh[k] = h_rel 
        else: # use a relative step 
            dh[k] = h_rel*x0[k]

    # Initial point 
    time0 = time.time()
    f0 = agg_fun(fhandle(x0)) 
    time1 = time.time()

    # expected time until calculations are done 
    sec_per_eval = time1-time0 
    evals = K + K*(K+1)//2 
    tot_time_secs = sec_per_eval * evals 
    if tot_time_secs > 5.0: # if we are slow, provide an ETA for the user 
        print(f'Computing numerical Hessian, expect approx. {tot_time_secs:5.2f} seconds (for {evals} criterion evaluations)')

    # Evaluate single forward steps
    for k in range(K): 
        x1 = np.copy(x0) 
        x1[k] = x0[k] + dh[k] 
        f1[k] = agg_fun(fhandle(x1))

    # Double forward steps
    for k in range(K): 
        for j in range(k+1): # only loop to the diagonal!! This is imposing symmetry to save computations
            
            # 1. find the new point (after double-stepping) 
            x2 = np.copy(x0) 
            if k==j: # diagonal steps: only k'th entry is changed, taking two steps 
                x2[k] = x0[k] + dh[k] + dh[k] 
            else:  # we have taken both a step in the k'th and one in the j'th directions 
                x2[k] = x0[k] + dh[k] 
                x2[j] = x0[j] + dh[j]  

            # 2. compute function value 
            f2[k,j] = agg_fun(fhandle(x2))
            
            # 3. fill out above the diagonal ()
            if j < k: # impose symmetry  
                f2[j,k] = f2[k,j]

    hess = np.empty((K,K))
    for k in range(K): 
        for j in range(K): 
            hess[k,j] = ((f2[k,j] - f1[k]) - (f1[j] - f0)) / (dh[k] * dh[j])

    return hess 

def centered_grad(f, x0: np.ndarray, h:float=1.49e-08) -> np.ndarray:
    '''centered_grad: numerical gradient calculator
    Args.
        f: function handle taking *one* input, f(x0). f can return a vector. 
        x0: P-vector, the point at which to compute the numerical gradient 

    Returns
        grad: N*P matrix of numericalgradients. 
    '''
    assert x0.ndim == 1, f'Assumes x0 is a flattened array'
    P = x0.size 

    # evaluate f at baseline 
    f0 = f(x0)
    N = f0.size

    # intialize output 
    grad = np.zeros((N, P))
    for i in range(P): 

        # initialize the step vectors 
        x1 = x0.copy()  # forward point
        x_1 = x0.copy() # backwards 

        # take the step for the i'th coordinate only 
        if x0[i] != 0: 
            x1[i] = x0[i]*(1.0 + h)  
            x_1[i] = x0[i]*(1.0 - h)
        else:
            # if x0[i] == 0, we cannot compute a relative step change, 
            # so we just take an absolute step 
            x1[i] = h
            x_1[i] = -h
        
        step = x1[i] - x_1[i] # the length of the step we took 
        grad[:, i] = ((f(x1) - f(x_1))/step).flatten()

    return grad