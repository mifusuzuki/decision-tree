#############################################################################
# 60012: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit(), predict() and prune() methods of
# DecisionTreeClassifier. You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import math

class Split():
    def __init__(self, col, val):
        self.col = col
        self.val = val
    

class Node():
    def __init__(self, left, right, split):
        self.is_leaf = False
        self.split = split
        self.left_child = left # less than val
        self.right_child = right # greater or equal to val
    
    def get_next_node(self, single_row):
        if single_row[self.split.col] < self.split.val:
            return self.left_child
        return self.right_child

class Leaf():
    def __init__(self, label):
        self.is_leaf = True
        self.label = label

class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.is_trained = False
        self.tree_root = None
    
    def calc_entropy(self, y):
        
        entropy = 0
        unique_label, freq = np.unique(y, return_counts=True)
        total = sum(freq)
        #print(unique_label, freq)
        #print(freq, total)
        for i, (label, freq) in enumerate(zip(unique_label, freq)):
            entropy += -(freq/total)*math.log(freq/total, 2)
        
        return entropy
        
    def partition(self, x, y, split):
        col = split.col
        val = split.val
        
        # partition
        #left_x = x[x[:, col] < val, :] # using masking
        #print(left_x)
        #right_x = x[x[:, col] >= val, :]
        #print(right_x)
        
        # left
        indices = np.where(x[:,col] < val)
        left_x = x[indices]
        left_y = y[indices]

        # right
        indices = np.where(x[:,col] >= val)
        right_x = x[indices]
        right_y = y[indices]
        
        return left_x,left_y, right_x, right_y

    def calc_info_gain(self, x, y, split):
        
        left_x,left_y, right_x, right_y = self.partition(x, y, split)
        
        # calc entropy
        left_entropy = self.calc_entropy(left_y)
        #print('left entropy = ', left_entropy)
        right_entropy = self.calc_entropy(right_y)
        #print('right entropy = ', right_entropy)
        original_entropy = self.calc_entropy(y)
        #print('original entropy = ', original_entropy)
        
        # calc information gain
        
        average_of_left_and_right_entropy = len(left_y)/len(y)*left_entropy + len(right_y)/len(y)*right_entropy
        info_gain = original_entropy - average_of_left_and_right_entropy
        
        
        return info_gain
        
    
    def find_optimal_split(self, x, y):
        
        optimal_split = None
        max_info_gain = 0
        seen = {}
        
        for row in range(len(x)):
            for col in range(len(x[0])):
                val = x[row][col]
                if col in seen and val in seen[col]:
                    continue
                if col not in seen:
                    seen[col] = []
                
                seen[col].append(val) # mark as seen
                split = Split(col, val) # create split
                info_gain = self.calc_info_gain(x, y, split)
                
                #print('info gain = ', info_gain, ' col = ', col, ' val = ', val)
                
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    optimal_split = split
        
        return max_info_gain, optimal_split
                
        
    def build_tree(self, x, y):
        
        info_gain, split = self.find_optimal_split(x, y)
        
        if info_gain == 0:
            #print('Leaf reached')
            return Leaf(y[0]) # Returns the first element since they are all the same - IS THIS TRUE THO??
        
        #print('optimal info gain = ', info_gain, ' col = ', split.col, ' val = ', split.val)
        left_x,left_y, right_x, right_y = self.partition(x, y, split)
        
        left_child = self.build_tree(left_x, left_y)
        right_child = self.build_tree(right_x, right_y)
        
        return Node(left_child, right_child, split)
      
    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################    
        
        # build a tree and assign the returned root to self.tree_root
        self.tree_root = self.build_tree(x, y)
  
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        
    def find_label(self, instance, cur_node):
        
        if cur_node.is_leaf:
            return cur_node.label
        next_node = cur_node.get_next_node(instance)
        return self.find_label(instance, next_node)
    
    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        
        # Investigate one row at a time
        for i, row in enumerate(x):
            predictions[i] = self.find_label(row, self.tree_root)
    
        # remember to change this if you rename the variable
        return predictions
        
    def prediction_accuracy(self, y_gold, y_prediction):
    
        assert len(y_gold) == len(y_prediction)
    
        try:
            return np.sum(y_gold == y_prediction) / len(y_gold)
        except ZeroDivisionError:
            return 0
        
        return correct/total

    def prune(self, x_val, y_val):
        """ Post-prune your DecisionTreeClassifier given some optional validation dataset.

        You can ignore x_val and y_val if you do not need a validation dataset for pruning.

        Args:
        x_val (numpy.ndarray): Instances of validation dataset, numpy array of shape (L, K).
                           L is the number of validation instances
                           K is the number of attributes
        y_val (numpy.ndarray): Class labels for validation dataset, numpy array of shape (L, )
                           Each element in y is a str 
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")


        #######################################################################
        #                 ** TASK 4.1: COMPLETE THIS METHOD **
        #######################################################################
       


