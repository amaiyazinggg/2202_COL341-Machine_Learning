from typing import List
import numpy as np
import pandas as pd
import qpsolvers
import kernel

class Trainer:
    def __init__(self,kernel,C=None,**kwargs) -> None:
        self.kernel = kernel
        self.kwargs = kwargs
        self.C=C
        self.support_vectors:List[np.ndarray] = []
        self.support_vector_y = None
        self.support_vector_alphas = None
        self.support_vector_km = None
        
    def load_data(self, train_data_path):
        df = pd.read_csv(train_data_path)
        df = df.iloc[:, 1:]
        x = df.loc[:, df.columns != 'y']
        if 'y' in df:
            y = df['y']
            y = y.to_numpy()
            y[y == 0] = -1
        else:
            y = np.array([])
        x = x.to_numpy()
        return x, y
    
    def fit(self, train_data_path:str)->None:
        #TODO: implement
        #store the support vectors in self.support_vectors
        
        x, y = self.load_data(train_data_path)

        n = len(y)
        
        km = self.kernel(x, **self.kwargs)
        y_prod = np.outer(y, y)
        
        P = y_prod*km
        q = -1*np.ones((n, 1))
        G = np.vstack((np.eye(n) * -1,np.eye(n)))
        h = np.hstack((np.zeros(n), np.ones(n) * self.C))
        A = y.reshape(1,-1)
        b = np.zeros(1)
 
        alphas = qpsolvers.solve_qp(P, q, G, h, A, b,solver="ecos")

        sv = (alphas > 1e-4).flatten()
        
        self.support_vectors = x[sv]
        self.support_vector_y = y[sv].reshape(-1,1)
        self.support_vector_alphas = alphas[sv].reshape(-1,1)
        self.support_vector_km = self.kernel(self.support_vectors, Z = self.support_vectors, **self.kwargs)
        
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels as a numpy array of dimension n_samples on test data
        x_test, y_test = self.load_data(test_data_path)
        
        b = self.support_vector_y - np.sum(self.support_vector_km * self.support_vector_alphas * self.support_vector_y, axis=0)
        b = np.sum(b) / b.size

        gx = np.sum(self.kernel(self.support_vectors, Z = x_test, **self.kwargs) * self.support_vector_alphas * self.support_vector_y, axis=0) + b
        predictions = np.sign(gx)
        
        predictions[predictions == -1] = 0
        
        return predictions
        
    def newfit(self, x, y)->None:
        #TODO: implement
        #store the support vectors in self.support_vectors
        n = len(y)
        
        km = self.kernel(x, **self.kwargs)
        y_prod = np.outer(y, y)
        
        P = y_prod*km
        A = y.reshape(1,-1)
        b = np.zeros(1)
        G = np.vstack((np.eye(n) * -1,np.eye(n)))
        h = np.hstack((np.zeros(n), np.ones(n) * self.C))
        q = -1*np.ones((n, 1))
 
        alphas = qpsolvers.solve_qp(P, q, G, h, A, b,solver="ecos")

        sv = (alphas > 1e-4).flatten()
        
        self.support_vectors = x[sv]
        self.support_vector_y = y[sv].reshape(-1,1)
        self.support_vector_alphas = alphas[sv].reshape(-1,1)
        self.support_vector_km = self.kernel(self.support_vectors, Z = self.support_vectors, **self.kwargs)
        
        pass
    
    def newpredict(self, x_test, y)->np.ndarray:
        #TODO: implement
        #Return the predicted labels as a numpy array of dimension n_samples on test data
        
        b = self.support_vector_y - np.sum(self.support_vector_km * self.support_vector_alphas * self.support_vector_y, axis=0)
        b = np.sum(b) / b.size

        gx = np.sum(self.kernel(self.support_vectors, Z = x_test, **self.kwargs) * self.support_vector_alphas * self.support_vector_y, axis=0) + b
        
        return gx
