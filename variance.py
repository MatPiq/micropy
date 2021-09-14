import numpy as np

class VarianceEstimator:
    
    @staticmethod
    def regular_variance(X: np.array, SSR)->tuple:

        """Use SSR and x array to calculate different regular variance.

        Args:
            SSR (float): SSR
            x (np.array): Array of independent variables.
        Raises:
            Exception: [description]
        Returns:
            tuple: [description]
        """
        N = X.shape[0]
        K = X.shape[1]
        sigma = SSR / (N - K)  
        cov =  sigma*la.inv(X.T@X)
        se =  np.sqrt(cov.diagonal()).reshape(-1, 1)
        return sigma, cov, se
    
    def perm(Q_T: np.array, A: np.array, t=0) -> np.array:
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
    # We can infer t from the shape of the transformation matrix.
        if t==0:
            t = Q_T.shape[1]

        # Initialize the numpy array
        Z = np.array([[]])
        Z = Z.reshape(0, A.shape[1])

        # Loop over the individuals, and permutate their values.
        for i in range(int(A.shape[0]/t)):
            Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
        return Z