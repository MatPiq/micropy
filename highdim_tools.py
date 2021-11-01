import numpy as np




def kfoldcv():
    pass

def normalize_X(X):
    """Normalizes an array to mean 0 and unit varaince

    Args:
        X (np.array): NxP feature matrix

    Returns:
        np.array: normalized NxP feature matrix
    """
    return (X - X.mean(axis = 0) / X.std(axis = 0))

def polynomial_features():
    pass

def bickel_ritov_tsybakov_rule():
    pass