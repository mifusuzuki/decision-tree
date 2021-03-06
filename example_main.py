##############################################################################
# 60012: Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np

from classification import *
from improvement import train_and_predict
from dataset import Dataset

if __name__ == "__main__":
    
    """
    print("Loading the training dataset...");
    x = np.array([
            [5,7,1],
            [4,6,2],
            [4,6,3], 
            [1,3,1], 
            [2,1,2], 
            [5,2,6]
        ])
    
    y = np.array(["A", "A", "A", "C", "C", "C"])
    
    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x, y)

    print("Loading the test set...")
    
    x_test = np.array([
            [1,6,3], 
            [0,5,5], 
            [1,5,0], 
            [2,4,2]
        ])
    
    y_test = np.array(["A", "A", "C", "C"])
    
    print("Making predictions on the test set...")
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))
    
    classes = ["A", "C"];
    
    print("Pruning the decision tree...")
    x_val = np.array([
                [6,7,2],
                [3,1,3]
            ])
    y_val = np.array(["A", "C"])
                   
    classifier.prune(x_val, y_val)
    
    print("Making predictions on the test set using the pruned decision tree...")
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))

    print("Making predictions on the test set using the improved decision tree...")
    predictions = train_and_predict(x, y, x_test, x_val, y_val)
    print("Predictions: {}".format(predictions))
    """
    print("Loading the training dataset...\n");
    
    train_full = Dataset()
    tf_x, tf_y = train_full.load('./data/train_full.txt')
    train_sub = Dataset()
    ts_x, ts_y = train_sub.load('./data/train_sub.txt')
    
    # Studying train_full.txt
    print('tf_x.shape = ', tf_x.shape)
    print('tf_y.shape = ', tf_y.shape)
    tf_labels, tf_freq = np.unique(tf_y, return_counts=True)
    print('labels = ', tf_labels, '\nfreq   = ', tf_freq)
    print('range of each attributes = \n', np.ptp(tf_x, axis=0))
    print('min of each attributes = \n', np.min(tf_x, axis=0))
    print('max of each attributes = \n', np.max(tf_x, axis=0))
    print('')
    
    # Studying train_sub.txt
    print('ts_x.shape = ', ts_x.shape)
    print('ts_y.shape = ', ts_y.shape)
    ts_labels, ts_freq = np.unique(ts_y, return_counts=True)
    print('labels = ', ts_labels, '\nfreq   = ', ts_freq)
    print('')
    
    # Loading noisy.txt
    noisy = Dataset()
    noisy_x, noisy_y = noisy.load('./data/train_noisy.txt')
    print('noisy_x.shape = ', noisy_x.shape)
    print('noisy_y.shape = ', noisy_y.shape)
    noisy_labels, noisy_freq = np.unique(noisy_y, return_counts=True)
    print('labels = ', noisy_labels, '\nfreq   = ', noisy_freq)
    print('')
    
    # 2.1
    # Loading toy.txt
    toy = Dataset()
    toy_x, toy_y = toy.load('./data/toy.txt')
    temp_classifier = DecisionTreeClassifier()
    #temp_split = Split(0, 4)
    #temp_classifier.calc_information_gain(toy_x, toy_y, temp_split)
    temp_classifier.build_tree(toy_x, toy_y)
    
    # 2.2
    # Loading test.txt
    test = Dataset()
    test_x, test_y = train_full.load('./data/test.txt')
    # Loading train_sub.txt
    train_sub = Dataset()
    ts_x, ts_y = train_sub.load('./data/train_sub.txt')
    temp_classifier = DecisionTreeClassifier()
    temp_classifier.fit(ts_x, ts_y)
    print(temp_classifier.is_trained)
    prediction = temp_classifier.predict(test_x)
    print('gold standard:\n')
    print(test_y)
    print('')
    print('prediction:\n')
    print(prediction)
    print('accuracy = ', temp_classifier.prediction_accuracy(test_y, prediction))
    
    