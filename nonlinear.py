import numpy as np
from numpy.core.fromnumeric import size
import scipy.stats as st
from numpy import linalg as la
import pandas as pd 
from scipy import optimize
from tabulate import tabulate
import abc



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

    def predict(self, X_new):
        assert self.res
        return X_new @ self.res['theta']

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
        N = y.size

        # The objective function is the average of q(), 
        # but Q is only a function of one variable, theta, 
        # which is what minimize() will expect
        Q = lambda theta: np.mean(self.criterion(theta, y, X))

        # call optimizer
        result = optimize.minimize(Q, theta0, options=options, **kwargs)
        
        cov, se = self.variance(self.criterion, y, X, result, cov_type)   

        # collect output in a dict 
        self.res = {
            'theta': result.x,
            'se':       se,
            't': result.x / se,
            'cov':      cov,
            'success':  result.success, # bool, whether convergence was succesful 
            'nit':      result.nit, # no. algorithm iterations 
            'nfev':     result.nfev, # no. function evaluations 
            'fun':      result.fun # function value at termination 
        }
        return self

    def variance(self,
        q, # function taking three inputs: q(theta, y, x) 
        y: np.ndarray, 
        x: np.ndarray, 
        result: dict, 
        cov_type: str
        ) -> tuple:
        """Calculates the variance for the likelihood function.

        Args:
            >> q: Function taking three inputs, q(theta,y,x), where we minimize the mean of q.
            >> y (np.ndarray): Dependent variable.
            >> x (np.ndarray): Independent variables.
            >> result (dict): Output from the function estimate().
            >> cov_type (str): Type of calculation to use in estimation.

        Returns:
            tuple: Returns the variance-covariance matrix and standard errors.
        """
        assert cov_type in ['hessian', 'outer product', 'sandwich']
        
        N = y.size
        if x.size == 3:
            N,K,J = x.shape
        elif x.size == 2:
            N,K = x.shape
       
        thetahat = result.x
        P = thetahat.size

        # numerical gradients 
        f_q = lambda theta : q(theta,y,x) 
        s = self.centered_grad(f_q, thetahat)

        # "B" matrix 
        B = (s.T@s)/N 
        
        # cov: P*P covariance matrix of theta 
        if cov_type == 'hessian':
            A_inv = result.hess_inv
            cov = 1/N * A_inv
        elif cov_type == 'outer product':
            cov = 1/N * la.inv(B)
        elif cov_type == 'sandwich':
            A_inv = result.hess_inv
            cov = 1/N * (A_inv @ B @ A_inv)

        # se: P-vector of std.errs. 
        se = np.sqrt(np.diag(cov))

        return cov, se

    def centered_grad(self, f, x0: np.ndarray, h:float=1.49e-08) -> np.ndarray:
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
        print(title)
        return tab 

class BinaryResponse(MestimatorBase):

    def __init__(self, estimator:str='logit'):
        assert estimator in ['logit', 'probit'],\
        f'Estimator must be one of logit, probit. Got {estimator}'
        self.estimator = estimator

    def starting_values(self, y, X) -> np.array:

        if self.estimator == 'logit':
            return 4*(la.inv(X.T@X)@(X.T@y))
        
        if self.estimator == 'probit':
            return 2.5*(la.inv(X.T@X)@(X.T@y))
        
    def loglikelihood(self, theta, y, X):
        if self.estimator == 'logit':
            Gxb = st.logistic.cdf(X @ theta)
        
        if self.estimator == 'probit':
            Gxb = st.norm.cdf(X @ theta)

         # we cannot take the log of 0.0
        Gxb = np.fmax(Gxb, 1e-8)    # truncate below at 0.00000001 
        Gxb = np.fmin(Gxb, 1.-1e-8) # truncate above at 0.99999999

        ll = (y == 1) * np.log(Gxb) + (y == 0) * np.log(1.0 - Gxb)# Fill in 
        return ll

    def sim_data(self, theta, N):
        beta = theta

        K = theta.size 
        assert K>1, f'Not implemented for constant-only'
        
        # 1. simulate x variables, adding a constant 
        oo = np.ones((N,1))
        xx = np.random.normal(size=(N,K-1))
        x = np.hstack([oo, xx]);
     
        # 2.a draw error terms 
        uniforms = np.random.uniform(size=(N,))
        u = st.norm.ppf(uniforms)

        # 2.b compute latent index 
        ystar = x@beta + u
        
        # 2.b compute observed y (as a float)
        y = (ystar>=0).astype(float)

        # 3. return 
        return y, x

class DiscreteRespons(MestimatorBase):

    def __init__(self, estimator:str='mlogit'):
        assert estimator in ['mlogit', 'clogit'],\
        f'Estimator must be one of mlogit, clogit. Got {estimator}'
        self.estimator = estimator

    def starting_values(self,_,X):
        return np.zeros(X.shape[2])
    
    def loglikelihood(self, theta, y, X) -> np.array:
        assert theta.ndim == 1 
        N = X.shape[0]

        # deterministic utility 
        v = self.util(theta, X) # Fill in (use util function)

        # denominator 
        denom = np.log(np.exp(v).sum(axis=1)) # Fill in
        assert denom.ndim == 1 # make sure denom is 1-dimensional so that we can subtract it later 
        # utility at chosen alternative 
        v_i = v[range(N), y]# Fill in evaluate v at cols indicated by y 

        # likelihood 
        ll_i = v_i - denom # Fill in 
        assert ll_i.ndim == 1 # we should return an (N,) vector 

        return ll_i 

    def util(self, theta, x, MAXRESCALE:bool=True): 
        '''util: compute the deterministic part of utility, v, and max-rescale it
        Args. 
            theta: (K,) vector of parameters 
            x: (N,J,K) matrix of covariates 
            MAXRESCALE (optional): bool, we max-rescale if True (the default)
        
        Returns
            v: (N,J) matrix of (deterministic) utility components
        '''
        assert theta.ndim == 1 
        # deterministic utility 
        v = x @ theta # Fill in 

        if MAXRESCALE: 
            # subtract the row-max from each observation
            # keepdims maintains the second dimension, (N,1), so broadcasting is successful
            v -= np.max(v, axis=1, keepdims=True) # Fill in
        
        return v

    def choice_prob(self, theta, x):
        '''choice_prob(): Computes the (N,J) matrix of choice probabilities 
        Args. 
            theta: (K,) vector of parameters 
            x: (N,J,K) matrix of covariates 
        
        Returns
            ccp: (N,J) matrix of probabilities 
        '''
        assert theta.ndim == 1, f'theta should have ndim == 1, got {theta.ndim}'        
        # deterministic utility 
        v = np.exp(self.util(theta, x)) 
        # denominator 
        denom =  v.sum(axis = 1, keepdims=True)
        assert denom.ndim == 2 # denom must be (N,1) so we can divide an (N,J) matrix with it without broadcasting errors
        
        # Conditional choice probabilites
        ccp = v / denom # Fill in 
        
        return ccp

    def sim_data(N: int, theta: np.ndarray, J: int) -> tuple:
        """Takes input values N and J to specify the shape of the output data. The
        K dimension is inferred from the length of theta. Creates a y column vector
        that are the choice that maximises utility, and a x matrix that are the 
        covariates, drawn from a random normal distribution.

        Args:
            N (int): Number of households.'
            J (int): Number of choices.
            theta (np.ndarray): The true value of the coefficients.

        Returns:
            tuple: y,x
        """
        K = theta.size
        
        # 1. draw explanatory variables 
        x = np.random.normal(size=(N,J,K)) # Fill in, use np.random.normal(size=())

        # 2. draw error term 
        uni = np.random.uniform(size=(N,J)) # Fill in random uniforms
        e = st.genextreme.ppf(uni, c=0) # Fill in: use inverse extreme value CDF

        # 3. deterministic part of utility (N,J)
        v = x @ theta # Fill in 

        # 4. full utility 
        u = v + e # Fill in 
        
        # 5. chosen alternative
        # Find which choice that maximises value: this is the discrete choice 
        y = np.argmax(u, axis = 1) # Fill in, use np.argmax(axis=1)
        assert y.ndim == 1 # y must be 1D
        
        return y,x