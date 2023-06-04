import numpy as np

# Do not change function signatures
#
# input:
#   X is the input matrix of size n_samples x n_features.
#   pass the parameters of the kernel function via kwargs.
# output:
#   Kernel matrix of size n_samples x n_samples 
#   K[i][j] = f(X[i], X[j]) for kernel function f()

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    Z = np.array([])
    
    for key, values in kwargs.items():
        if key == "gamma":
            gamma = values
        if key == "Z":
            Z = values
            
    if Z.size == 0:
        n = X.shape[0]
        m = X.shape[0]
        
        xx = np.dot(np.sum(X, 1).reshape(n, 1), np.ones((1, m)))
        zz = np.dot(np.sum(X, 1).reshape(m, 1), np.ones((1, n)))
        kernel_matrix = np.exp(-gamma*(xx - zz.T))
    else:
        n = X.shape[0]
        m = Z.shape[0]
        
        xx = np.dot(np.sum(X, 1).reshape(n, 1), np.ones((1, m)))
        zz = np.dot(np.sum(Z, 1).reshape(m, 1), np.ones((1, n)))
        kernel_matrix = np.exp(-gamma*(xx - zz.T))
        
    return kernel_matrix
    
def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    Z = np.array([])
    
    for key, values in kwargs.items():
        if key == "Z":
            Z = values
        if key == "r":
            r = values
        if key == "gamma":
            gamma = values
            
    if Z.size == 0:
        kernel_matrix = np.tanh(gamma*np.dot(X, X.T)+r)
    else:
        kernel_matrix = np.tanh(gamma*np.dot(X, Z.T)+r)
        
    return kernel_matrix
    
def rbf(X: np.ndarray, **kwargs)-> np.ndarray:
    assert X.ndim == 2
    Z = np.array([])
    
    for key, values in kwargs.items():
        if key == "gamma":
            gamma = values
        if key == "Z":
            Z = values
            
    if Z.size == 0:
        n = X.shape[0]
        m = X.shape[0]
        
        xx = np.dot(np.sum(np.power(X, 2), 1).reshape(n, 1), np.ones((1, m)))
        zz = np.dot(np.sum(np.power(X, 2), 1).reshape(m, 1), np.ones((1, n)))
        kernel_matrix = np.exp(-gamma*(xx + zz.T - 2 * np.dot(X, X.T)))
    else:
        n = X.shape[0]
        m = Z.shape[0]
        
        xx = np.dot(np.sum(np.power(X, 2), 1).reshape(n, 1), np.ones((1, m)))
        zz = np.dot(np.sum(np.power(Z, 2), 1).reshape(m, 1), np.ones((1, n)))
        kernel_matrix = np.exp(-gamma*(xx + zz.T - 2 * np.dot(X, Z.T)))
        
    return kernel_matrix

def linear(X: np.ndarray, **kwargs)-> np.ndarray:
    assert X.ndim == 2
    Z = np.array([])
    
    for key, values in kwargs.items():
        if key == "Z":
            Z = values
            
    if Z.size == 0:
        kernel_matrix = np.dot(X, X.T)
    else:
        kernel_matrix = np.dot(X, Z.T)
        
    return kernel_matrix

def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    Z = np.array([])
    
    for key, values in kwargs.items():
        if key == "Z":
            Z = values
        if key == "d":
            d = values
            
    if Z.size == 0:
        kernel_matrix = np.power(np.dot(X, X.T)+1, d)
    else:
        kernel_matrix = np.power(np.dot(X, Z.T)+1, d)
        
    return kernel_matrix
