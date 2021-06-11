import numpy as np
from collections import Counter
from itertools import repeat
from multiprocessing import Pool


class Node:
    def __init__(self, parent, impurity, indices, depth, children=None, \
        feature=None, splitter=None, gain=None):
        """a class node that represents a node in the tree"""
        self.parent = parent        # parent of node
        self.impurity = impurity    # entropy, gini, mse, or mae of node
        self.indices = indices      # indices of samples in the node
        self.depth = depth          # depth of node
        self.children = children    # children of node after split
        self.feature = feature      # feature used for split
        self.splitter = splitter    # value of the feature used for split
        self.gain = gain            # information gain or variance reduction gain after split
        

class Tree:
    def __init__(self, criterion, max_depth, max_leaves, max_features,\
        min_sample_split):
        """base Tree class"""
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_features = max_features
        self.min_sample_split = min_sample_split

    def fit(self, x, y):
        self.y = y
        m = x.shape[0]

        impurity = self._get_impurity(y, m)     # get gini or entorpy or mse or l1 of root node
        self.root = Node(None, impurity, np.arange(m), 0)      # root node
        impure_leaves = [self.root]     # nodes with gini or entorpy greater than zero
        
        while impure_leaves:
            node = impure_leaves.pop()
            if not node.children and node.impurity:      # if not split & entropy > 0
                if self.max_depth and node.depth == self.max_depth:
                    continue    # pass if max_depth is reached
                elif self.min_sample_split and len(node.indices) < self.min_sample_split:
                    continue    # pass if samples in node is less than min_sample_split
                else:
                    self._split(x, y, node)
                    if node.children:       # else node wasn't split because no gain found
                        impure_leaves.extend(node.children)

        if self.max_leaves and self.max_leaves > 2: # if True prune the tree
            leaves = self._get_leaves(self.root)
            while len(leaves) > self.max_leaves:
                least_gain = np.inf
                least_gain_leaves = None    # placeholder for terminal node siblings
                for leaf in leaves:
                    siblings = leaf.parent.children
                    # check if leaf's sibling is terminal node
                    # check both siblings since index of "leaf" not known
                    if not siblings[0].children and \
                        not siblings[1].children:
                        if siblings[0].parent.gain < least_gain:
                            least_gain = siblings[0].parent.gain
                            least_gain_leaves = siblings

                least_gain_leaves[0].parent.children = None   # cut children from tree
                leaves.remove(least_gain_leaves[0])
                leaves.remove(least_gain_leaves[1])
                leaves.append(least_gain_leaves[0].parent)

    def _split(self, x, y, node):
        """finds the best split for a node
        """
        x = x[node.indices]
        y = y[node.indices]
        # prepare the features to be considered during split
        if self.max_features:
            feature_indices = np.random.choice(x.shape[1], self.max_features)
        else:
            feature_indices = range(x.shape[1])

        best_gain = 0          # placeholder for best gain across features
        best_feature = None    # placeholder for feature with best gain
        splitter = None        # value of splitter in best feature
        left_node = {
            'parent' : node,
            'impurity' : None,
            'indices' : None,
            'depth' : node.depth + 1
        }
        right_node = left_node.copy()
        
        for i in feature_indices:      # iterate across the feature indices
            feature_x = x[:, i]
            # sort the unique feature_x values to be used for splitting points
            # convert feature_x to int if it is a continuous feature
            x_uniques = np.sort(np.unique(feature_x.astype(int)))
            values = x_uniques[1:]      # skip first value since "<" would make an empty left node

            for value in values:
                impurity_left, impurity_right, gain = \
                    self._get_gain(feature_x, y, node.impurity, value)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    splitter = value
                    left_node['impurity'] = impurity_left
                    left_node['indices'] = node.indices[feature_x < splitter]
                    right_node['impurity'] = impurity_right
                    right_node['indices'] = node.indices[feature_x >= splitter]

        if best_gain > 0:       # else the node will be a leaf node
            node.feature = best_feature
            node.gain = best_gain
            node.splitter = splitter
            node.children = [Node(**left_node), Node(**right_node)]

    def _get_gain(self, feature_x, y, parent_impurity, value):
        """calculates gain of given samples
        
        Arguments:
        feature_x -- feature of x to be split
        y -- vector of labels for samples
        parent_impurity -- impurity of node before split
        value -- value of feature x to be used as split point

        Returns:
        {impurity_left, impurity_right, gain} 
            -- impurity of the samples left, right child nodes and gain
        """
        m = len(feature_x)
        len_left = np.sum(feature_x < value)
        # calculate gini or entropy of splitted nodes
        impurity_left = self._get_impurity(y[feature_x < value], len_left)
        impurity_right = self._get_impurity(y[feature_x >= value], m - len_left)              

        gain = parent_impurity - impurity_left * len_left / m - impurity_right * (m - len_left) / m

        return impurity_left, impurity_right, gain
             
    def _get_leaves(self, node):
        """returns all terminal leaves of tree"""
        leaves = []
        if node.children:
            for child in node.children:
                leaves.extend(self._get_leaves(child))
        else:
            leaves.append(node)

        return leaves

    def predict(self, x):
        predictions = []
        for instance in x:
            predictions.append(self._find_leaf(instance, self.root))

        return predictions

    def _find_leaf(self, instance, node):
        """recursively find terminal leaf based on x instance's 
        feature values"""
        if node.children:           # check if the node is split
            if instance[node.feature] < node.splitter:
                # look for values on the left child
                prediction = self._find_leaf(instance, node.children[0])
            else:
                # look for values on the right child
                prediction = self._find_leaf(instance, node.children[1])
        else:
            leaf_values = self.y[node.indices]      # label values at node
            if isinstance(self, DecisionTreeRegressor):
                prediction = np.mean(leaf_values)
            else:
                prediction = Counter(leaf_values).most_common()[0][0]

        return prediction

class DecisionTreeRegressor(Tree):
    """Decision Tree regressor class"""
    def __init__(self, criterion='mse', max_depth=None,\
        max_leaves=None, max_features=None, min_sample_split=None):
        super().__init__(criterion, max_depth, max_leaves, \
            max_features, min_sample_split)

    def _get_impurity(self, y, _):
        """calculates mse or mae of given samples
        
        Arguments:
        y -- vector of labels for samples
        _ -- positional argument kept for consistency with _get_impurity
                method of DecisionTreeClassifier

        Returns:
        impurity -- impurity of the samples in terms of mse or mae
        """
        if self.criterion == 'mse':
            mean = np.mean(y)
            impurity = np.mean(np.square(y - mean))
        else:
            mode = Counter(y).most_common()[0][0]
            impurity = np.mean(np.abs(y - mode))

        return impurity


class DecisionTreeClassifier(Tree):
    """Decision Tree classifier class"""
    def __init__(self, criterion='entropy', max_depth=None,\
        max_leaves=None, max_features=None, min_sample_split=None):
        super().__init__(criterion, max_depth, max_leaves, \
            max_features, min_sample_split)

    def _get_impurity(self, y, m):
        """calculates entropy or gini of given samples
        
        Arguments:
        y -- vector of labels for samples
        m -- length of samples

        Returns:
        impurity -- impurity of the samples in terms of gini or entropy
        """
        classes = np.unique(y)
        probabilities = np.array([np.sum(y == label) / m for label in classes])
        if self.criterion == 'entropy':
            impurity =  -1 * np.sum(probabilities * np.log2(probabilities))
        else:
            impurity = 1 - np.sum(probabilities**2)

        return impurity
    

def test_regressor():
    x, y = load_boston(True)

    start_time = time.time()
    tree = DecisionTreeRegressor('mse', max_depth=10)
    tree.fit(x, y)
    print(mean_squared_error(tree.predict(x), y))
    print(f'Elapsed time: {time.time() - start_time}')

def test_classifier():
    x, y = load_iris(True)

    start_time = time.time()
    tree = DecisionTreeClassifier('gini', max_depth=10)
    tree.fit(x, y)
    print(sum(tree.predict(x) != y))
    print(f'Elapsed time: {time.time() - start_time}')

if __name__ == '__main__':
    import time
    import pytest
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston, load_iris

    test_regressor()
    test_classifier()

