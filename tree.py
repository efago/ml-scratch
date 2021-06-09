import numpy as np
from collections import Counter
from multiprocessing import Pool
from sklearn.datasets import load_iris


class Node:
    def __init__(self, parent, impurity, indices, depth, children=None, \
        feature=None, splitter=None, gain=None):
        """a class node that represents a node in the tree
        """
        self.parent = parent        # parent of node
        self.impurity = impurity    # entropy or gini of node
        self.indices = indices      # indices of samples in the node
        self.depth = depth          # depth of node
        self.children = children    # children of node after split
        self.feature = feature      # feature used for split
        self.splitter = splitter    # value of the feature used for split
        self.gain = gain            # information gain after split
        

class Tree:
    def __init__(self, criterion, max_depth, max_nodes, \
        max_features, min_sample_split, n_jobs):
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.n_jobs = n_jobs


class DecisionTree(Tree):
    def __init__(self, criterion='entropy', max_depth=None,\
        max_nodes=0, max_features=None, min_sample_split=None, n_jobs=1):
        super().__init__(criterion, max_depth, max_nodes, \
            max_features, min_sample_split, n_jobs)

    def fit(self, x, y):
        self.y = y
        m = x.shape[0]

        impurity = self._get_impurity(y, m)         # get gini or entorpy of root node
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
                    if node.children:       # else node wasn't split and no gain found
                        impure_leaves.extend(node.children)


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
            x_uniques = np.sort(np.unique(feature_x))   # sorted unique values for splitting
            m = len(feature_x)

            values = x_uniques[1:]      # skip first value since "<" would make an empty left node
            len_lefts = np.sum(feature_x.reshape(-1, 1) < values, axis=0)   # length of samples of left node

            if self.n_jobs > 1:
                with Pool(processes=self.n_jobs) as pool:
                    gains = pool.starmap(self._get_gain, zip(values, len_lefts))
            else:
                gains = []  # placeholder for dicts of (left_impurity, right_impurity, gain) of all splits
                for j in range(len(values)):
                    gains.append(self._get_gain(values[j], len_lefts[j]))

            for k, gain in enumerate(gains):
                if gain['gain'] > best_gain:
                    best_gain = gain['gain']
                    best_feature = i
                    splitter = values[k]
                    left_node['impurity'] = gain['left_impurity']
                    left_node['indices'] = node.indices[feature_x < splitter]
                    right_node['impurity'] = gain['right_impurity']
                    right_node['indices'] = node.indices[feature_x >= splitter]

        if best_gain > 0:       # else the node will be a leaf node
            node.feature = best_feature
            node.gain = best_gain
            node.splitter = splitter
            node.children = [Node(**left_node), Node(**right_node)]

    def _get_gain(self, feature_x, y, m):
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

    def predict(self, x):
        predictions = []
        for instance in x:
            predictions.append(self._find_leaf(instance, self.root))

        return predictions

    def _find_leaf(self, instance, node):
        if node.children:           # check if the node is split
            if instance[node.feature] < node.splitter:
                # look for values on the left child
                prediction = self._find_leaf(instance, node.children[0])
            else:
                # look for values on the right child
                prediction = self._find_leaf(instance, node.children[1])
        else:
            leaf_values = self.y[node.indices]      # label values at node
            prediction = Counter(leaf_values).most_common()[0][0]

        return prediction


if __name__ == '__main__':
    x = np.array([[31, 29, 27, 35, 28, 40, 39], [45, 47, 55, 53, 51, 50, 41]]).T
    y = np.array([1, 0, 0, 1, 0, 1, 0])
    x_iris, y_iris = load_iris(True)
    tree = DecisionTree('entropy', max_features=1)
    tree.fit(x_iris, y_iris)
    print(tree.predict(x_iris) - y_iris)