## Findings

1.1 Besides the number of instances, what is another main difference between train full.txt and train sub.txt?

While full version has similar number of instances for each class, sub version doesn't.
tf_labels = ['A' 'C' 'E' 'G' 'O' 'Q'] 
Freq =      [667 599 659 671 637 667]
ts_labels = ['A' 'C' 'E' 'G' 'O' 'Q']
Freq =      [ 94 187 129  21 113  56]

1.2 What kind of attributes are provided in the dataset (Binary? Categorical/Discrete? Integers? Real numbers?) What are the ranges for each attribute in train full.txt?

Real/Int
Attribute range = [10. 15. 10. 12. 14. 12. 14. 10. 12. 10. 11. 11. 12. 14. 11. 13.]
Attribute min =   [0. 0. 1. 0. 0. 2. 0. 0. 0. 3. 0. 4. 0. 1. 0. 1.]
Attribute max =   [10. 15. 11. 12. 14. 14. 14. 10. 12. 13. 11. 15. 12. 15. 11. 14.]

1.3 train noisy.txt is actually a corrupted version of train full.txt, where we have replaced the ground truth labels with the output of a simple automatic classifier. What proportion of labels in train noisy.txt is different than from those in train full.txt?(Note that the observations in both datasets are the same, although the ordering is different). Has the class distribution been affected? Specify which classes have a substantially larger or smaller number of examples in train noisy.txt compared to train full.txt.

Although all relevant labels appear as in train_full, the distribution has changed. Label 'G' in particular has been affected, with a difference of 49 (671 in train_full and 622 in train_noisy).
noisy_labels =  ['A' 'C' 'E' 'G' 'O' 'Q'] 
freq   =        [681 571 678 622 666 682]
Difference =    [14, 28, 19, 49, 29, 15]

2.1 Implementation of fit() in DecisionTreeClassifier

fit(x, y) calls build_tree(x, y), which first takes the entire entries and computes the most optimal split by iterating through each elements (i.e. splitting at val 4 on col 2). 

The max of information gain of each hypothetical split is tracked as well as storing already calculated split to skip such any duplicated cases (note split(col=1, val=2) and split(col=2, val=2) are not considered duplicate as their attributes differ). 

In order to find individual hypothetical information gain for all possible splits, calc_info_gain(x, y, split) is called (which calls partition(x, y, split) that returns entries spited according to a given split condition) and taking the returned hypothetical left branch and right branch that are returned, it calculates the respective entries. It then finally returns the information gain calculated by deducting the average entropy of the child branches from the original entropy.

build_tree() carries out post-order DFS by expanding the left most branches first then creates the right, then itself after which it returns itself. The last node returned by the top-most layer of the function is the root node, which is assigned to the classifier's self.tree_root.





