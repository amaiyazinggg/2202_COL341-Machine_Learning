import numpy as np
import pandas as pd
import cv2 as cv
import glob

class Node:
    def __init__(self, feature=None, threshold=None, left_child = None, right_child = None, info_gain=None, value=None):
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.info_gain = info_gain
        self.value = value
        
class DecisionTree_Gain:
    
    def __init__(self, min_samples_split=7, max_depth=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def calc_entropy(self, y):
        #np.bincount calcultes number of elements in each class
        number_of_each_class = np.bincount(np.array(y, dtype=np.int64))
        
        #These are the probabilities
        probabilities = number_of_each_class/len(y)

        entropy = 0
        
        for prob in probabilities:
            if prob > 0:
                #using the formula for entropy
                entropy -= np.log2(prob) * prob
                
        return entropy
    
    def information_gain(self, left, right, parent):
        p = len(parent)
        
        #calculate number of entries in the left and right child nodes
        l_prob = len(left)/p
        r_prob = len(right)/p
        
        #Using the formula for information gain
        gain = self.calc_entropy(parent) - r_prob * self.calc_entropy(right) - l_prob * self.calc_entropy(left)
        
        return gain
    
    def find_split(self, x, y):
        split_best = {}
        gain_best = -1
        
        rows = x.shape[0]
        columns = x.shape[1]
        
        for feature in range(columns):
            
            selected_column = x[:, feature]

            for threshold in np.unique(selected_column):
                
                data = np.concatenate((x, y.reshape(1, -1).T), axis=1)
                
                left_child = [vector for vector in data if vector[feature] <= threshold]
                left_child = np.array(left_child)
                
                right_child = [vector for vector in data if vector[feature] > threshold]
                right_child = np.array(right_child)

                if len(left_child) > 0 and len(right_child) > 0:
                    y = data[:, -1]
                    y_left = left_child[:, -1]
                    y_right = right_child[:, -1]

                    gain = self.information_gain(y_left, y_right, y)
                    if gain > gain_best:
                        split_best = {'feature_index': feature,'threshold': threshold,'df_left': left_child,'df_right': right_child,'gain': gain}
                        gain_best = gain
                        
        return split_best

    
    def find_maxcount(self, y):
        zero_count = 0
        one_count = 0
        
        for i in y:
            if i == 0:
                zero_count += 1
            else:
                one_count += 1
        
        if zero_count > one_count:
            return 0
        else:
            return 1
    
    def construct_tree(self, x, y, depth=0):
        rows = x.shape[0]
        columns = x.shape[0]
    
        if rows >= self.min_samples_split and depth <= self.max_depth:
            
            optimal_split = self.find_split(x, y)

            if optimal_split['gain'] > 0:
                left = self.construct_tree(optimal_split['df_left'][:, :-1], optimal_split['df_left'][:, -1], depth + 1)
                right = self.construct_tree(optimal_split['df_right'][:, :-1], optimal_split['df_right'][:, -1], depth=depth + 1)
                
                return Node(optimal_split['feature_index'], optimal_split['threshold'], left, right, optimal_split['gain'])

        return Node(value = self.find_maxcount(y))
    
    def fit(self, x, y):
        self.root = self.construct_tree(x, y)
        
    def predict_helper(self, my_node, x):
        #value is Not None only for leaf nodes
        if my_node.value != None:
            return my_node.value
        
        #take that particular feature of the entry
        compare_this = x[my_node.feature]
        
        #check on which side of the threshold it is
        if compare_this <= my_node.threshold:
            return self.predict_helper(my_node.left_child, x)
        else:
            return self.predict_helper(my_node.right_child, x)
        
    def predict(self, x):
        y_prediction = [self.predict_helper(self.root, i) for i in x]
        return y_prediction
        
        
        
class DecisionTree_Gini:
    
    def __init__(self, min_samples_split=7, max_depth=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def gini(self, y):
        number_of_each_class = np.bincount(np.array(y, dtype=np.int64))
        probabilities = number_of_each_class/len(y)
        prob_squared = np.square(probabilities)
        gini = 1 - np.sum(prob_squared)
        return gini
    
    def find_split(self, x, y):
        split_best = {}
        gini_best = 1
        
        rows = x.shape[0]
        columns = x.shape[1]
        
        for feature in range(columns):
            
            selected_column = x[:, feature]

            for threshold in np.unique(selected_column):
                
                data = np.concatenate((x, y.reshape(1, -1).T), axis=1)
                
                left_child = [vector for vector in data if vector[feature] <= threshold]
                left_child = np.array(left_child)
                
                right_child = [vector for vector in data if vector[feature] > threshold]
                right_child = np.array(right_child)

                if len(left_child) > 0 and len(right_child) > 0:
                    y = data[:, -1]
                    y_left = left_child[:, -1]
                    y_right = right_child[:, -1]

                    gini = (self.gini(y_left)*len(y_left) + self.gini(y_right)*len(y_right))/(len(y_left) + len(y_right))
                    if gini < gini_best:
                        split_best = {'feature_index': feature,'threshold': threshold,'df_left': left_child,'df_right': right_child, 'gini': gini}
                        gini_best = gini
                        
        return split_best

    
    def find_maxcount(self, y):
        zero_count = 0
        one_count = 0
        
        for i in y:
            if i == 0:
                zero_count += 1
            else:
                one_count += 1
        
        if zero_count > one_count:
            return 0
        else:
            return 1
    
    def construct_tree(self, x, y, depth=0):
        rows = x.shape[0]
        columns = x.shape[0]
    
        if rows >= self.min_samples_split and depth <= self.max_depth:
            
            optimal_split = self.find_split(x, y)

            if optimal_split['gini'] > 0:
                left = self.construct_tree(optimal_split['df_left'][:, :-1], optimal_split['df_left'][:, -1], depth + 1)
                right = self.construct_tree(optimal_split['df_right'][:, :-1], optimal_split['df_right'][:, -1], depth=depth + 1)
                
                return Node(optimal_split['feature_index'], optimal_split['threshold'], left, right, optimal_split['gini'])

        return Node(value = self.find_maxcount(y))
    
    def fit(self, x, y):
        self.root = self.construct_tree(x, y)
        
    def predict_helper(self, my_node, x):
        #value is Not None only for leaf nodes
        if my_node.value != None:
            return my_node.value
        
        #take that particular feature of the entry
        compare_this = x[my_node.feature]
        
        #check on which side of the threshold it is
        if compare_this <= my_node.threshold:
            return self.predict_helper(my_node.left_child, x)
        else:
            return self.predict_helper(my_node.right_child, x)
        
    def predict(self, x):
        y_prediction = [self.predict_helper(self.root, i) for i in x]
        return y_prediction