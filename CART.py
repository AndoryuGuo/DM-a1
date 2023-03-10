import numpy as np

from typing import Optional, Union, Tuple
from collections import OrderedDict

class Node:
    def __init__(self, split_feature: Union[str, int, None]=None, 
                 split_val: Union[str, float, None]=None, 
                 impurity: Optional[float]=0.0, 
                 left_child=None, 
                 right_child=None,
                 is_leaf: bool=False,
                 class_counts: np.ndarray=None) -> None:
        self.split_feature = split_feature
        self.split_val = split_val
        self.impurity = impurity
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf: bool = is_leaf
        self.class_counts: np.ndarray = class_counts
        
    def prediction(self, x: np.ndarray) -> int:
        if self.is_leaf:
            return np.argmax(self.class_counts)
        
        if x[self.split_feature] <= self.split_val:
            return self.left_child.prediction(x)
        
        return self.right_child.prediction(x)


class CART:
    def __init__(self, max_depth: int = None, min_samples_split=2, min_samples_leaf=1, max_leaves=None) -> None:
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.max_leaves: int = max_leaves
        self.n_leaves: int = 0
        self.n_features: int = 0
        self.n_classes: int = 0
        self._tree: Node = None

    def __grow_tree(self, X: np.ndarray, y: np.ndarray, depth=0):
        n_samples: int = X.shape[0]
        class_counts: np.ndarray = np.bincount(y, minlength=self.n_classes) # np.array([np.sum(y == c) for c in range(self.n_classes)])
        n_classes: int = np.count_nonzero(class_counts)

        # check if any stopping criterion reached
        if (n_samples < self.min_samples_split 
            or n_classes == 1 
            or (self.max_depth and depth >= self.max_depth)
            or (self.max_leaves and self.max_leaves < self.n_leaves + 2)):
            self.n_leaves += 1
            return Node(is_leaf=True, class_counts=class_counts, 
                        impurity=self.__compute_gini(class_counts)) # FIXME: complete node class
        
        # TODO: random selection of features
        split_feature: int
        split_val: float
        
        split_feature, split_val, impurity = self.find_best_candidate(X, y, class_counts)

        if split_feature == None:
            self.n_leaves += 1
            return Node(is_leaf=True, class_counts=class_counts, 
                        impurity=self.__compute_gini(class_counts))

        left_indices, right_indices = self.find_split_indices(X, split_feature, split_val)

        left_node = self.__grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_node = self.__grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        return Node(split_feature=split_feature, split_val=split_val, impurity=impurity,
                    left_child=left_node, right_child=right_node, class_counts=class_counts)

    def find_best_candidate(self, X, y, class_counts) -> Tuple[int, Union[str, float]]:
        # best candidate refers to best splitting feature and its corresponding splitting value
        n_samples: int = X.shape[0]
        best_gini_idx: float = float('inf')
        split_feature: int = None
        split_val: int = None

        # Get the sorted indices of X by each feature
        sorted_X_indices: np.ndarray = np.argsort(X, axis=0)

        for i in range(self.n_features):
            # Create count matrix
            # sorted_x_i = np.argsort(X[:, i])
            # Corresponding y sorted by i-th feature
            sorted_y: np.ndarray = y[sorted_X_indices[:,i]]
            left_bound: float = X[sorted_X_indices[0,i], i] # smallest value of i-th feature
            left_bound_cnts: np.ndarray = np.zeros(self.n_classes, dtype=int)
            left_bound_cnts[sorted_y[0]] = 1
        
            # Stores the counts of each class of each bin in X, the first bin is dummy
            bin_counts: OrderedDict = OrderedDict([(left_bound, np.zeros(self.n_classes, dtype=int))])  
            last_bin = left_bound # dummy bin

            for j in range(1, n_samples):
                x_ji: float = X[sorted_X_indices[j,i], i] # Feature value of X sorted by i-th feature
                y_j: int = sorted_y[j]
                
                if x_ji == left_bound:
                    left_bound_cnts[y_j] += 1
                    continue
                # Initialize the counts for new bin
                new_bin: float = (left_bound + x_ji) / 2
                bin_counts[new_bin] = bin_counts[last_bin] + left_bound_cnts # np.take(bin_counts[last_bin], np.arange(self.n_classes))

                last_bin = new_bin
                left_bound = x_ji
                left_bound_cnts = np.zeros(self.n_classes, dtype=int)
                left_bound_cnts[y_j] = 1

            # remove the dummy bin
            bin_counts.popitem(last=False)

            for bin, counts in bin_counts.items():
                # Compute gini index for each possible split
                gini_idx = self.compute_gini_index(counts, class_counts, y)

                if best_gini_idx > gini_idx:
                    best_gini_idx = gini_idx
                    split_feature = i
                    split_val = bin

        return split_feature, split_val, best_gini_idx

    def compute_gini_index(self, lte_class_counts: np.ndarray, class_counts: np.ndarray, y: np.ndarray):
        n_left_samples: int = lte_class_counts.sum()
        left_p: float = n_left_samples / len(y)
        right_p: float = (len(y) - n_left_samples) / len(y)
        gini_idx = (left_p * self.__compute_gini(lte_class_counts) 
                    + right_p * self.__compute_gini(class_counts - lte_class_counts))

        return gini_idx
    
    def __compute_gini(self, class_counts):
        return 1 - np.sum((class_counts / np.sum(class_counts)) ** 2)


    def find_split_indices(self, X: np.ndarray, split_feature: int, split_val: float):
        left_indices = X[:, split_feature] <= split_val 
        right_indices = X[:, split_feature] > split_val
        return left_indices, right_indices

    def fit(self, X, y) -> None:
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.n_labels = len(np.unique(y))
        self._tree = self.__grow_tree(X, y)

    def predict(self, X) -> int:
        pred_y: np.ndarray = np.array([self.__predict_instance(x) for x in X])
        return pred_y

    def __predict_instance(self, x):
        return self._tree.prediction(x)