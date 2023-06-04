from typing import List
import numpy as np
import pandas as pd
import kernel
from svm_binary import Trainer

class Trainer_OVA:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
        
    def load_train_data(self, train_data_path, class_num):
        df = pd.read_csv(train_data_path)
        df = df.iloc[:, 1:]
        x = df.loc[:, df.columns != 'y']
        x = x.to_numpy()
        
        if 'y' in df:
            y = df['y']
            y = y.to_numpy()
            y[y != class_num] = -1
            y[y == class_num] = 1
        else:
            y = np.array([])
            
        return x, y
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers
        self.svms = [Trainer(self.kernel, self.C, **self.kwargs) for i in range(self.n_classes)]
        pass
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms
        for i in range(self.n_classes):
            x, y = self.load_train_data(train_data_path, i+1)
            self.svms[i].newfit(x, y)
        pass
    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels
        for i in range(self.n_classes):
            x, y = self.load_train_data(test_data_path, i+1)
            if i == 0:
                predictions = self.svms[i].newpredict(x, y)
            else:
                predictions = np.vstack([predictions, self.svms[i].newpredict(x, y)])
                
        final_pred = []
        for i in range(len(x)):
            maxi = 1
            for j in range(self.n_classes):
                if predictions[j][i] > predictions[maxi-1][i]:
                    maxi = j+1
            final_pred.append(maxi)
        final_pred = np.array(final_pred)
            
        return final_pred
    
class Trainer_OVO:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
        
    def load_train_data(self, train_data_path, class1, class2):
        df = pd.read_csv(train_data_path)
        df = df.iloc[:, 1:]
        x = df.loc[:, df.columns != 'y']
        x = x.to_numpy()
        
        if 'y' in df:
            y = df['y']
            y = y.to_numpy()
            y[(y != class1) & (y != class2)] = 0
            y[y == class1] = 1
            y[y == class2] = -1
            ind = (y != 0).flatten()
            y = y[ind]
            x = x[ind]
        else:
            y = np.array([])
            
        return x, y
        
    def load_test_data(self, test_data_path):
        df = pd.read_csv(test_data_path)
        df = df.iloc[:, 1:]
        x = df.loc[:, df.columns != 'y']
        x = x.to_numpy()
        y = np.array([])
        
        return x, y
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers
        for i in range(self.n_classes):
            self.svms.append([])
            for j in range(self.n_classes):
                self.svms[i] = [Trainer(self.kernel, self.C, **self.kwargs) for j in range(self.n_classes)]
        pass
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if (i != j):
                    x, y = self.load_train_data(train_data_path, i+1, j+1)
                    self.svms[i][j].newfit(x, y)
        pass
    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels
        x, y = self.load_test_data(test_data_path)
        weights = np.zeros((self.n_classes, len(x)))
        
        for i in range(self.n_classes):
            for j in range(i+1, self.n_classes):
                midpred = self.svms[i][j].newpredict(x, y)
                for k in range(len(x)):
                    if midpred[k] > 0:
                        weights[i][k] = midpred[k]
                    else:
                        weights[j][k] = -midpred[k]
        
        final_pred = []
        for i in range(len(x)):
            maxi = 1
            for j in range(self.n_classes):
                if weights[j][i] > weights[maxi-1][i]:
                    maxi = j+1
            final_pred.append(maxi)
        final_pred = np.array(final_pred)
            
        return final_pred

