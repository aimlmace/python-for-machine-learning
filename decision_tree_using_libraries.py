import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from collections import Counter

class ID3:
    def __init__(self):
        self.tree = None
        self.class_names = None
   
    def fit(self, X, y, class_names):
        """Train the ID3 decision tree classifier with class names."""
        self.class_names = class_names
        self.tree = self._build_tree(X, y)
   
    def predict(self, X):
        """Predict class labels for samples in X using class names."""
        predictions = [self._predict_sample(self.tree, sample) for sample in X]
        return np.array([self.class_names[p] for p in predictions])
   
    def _entropy(self, y):
        """Calculate the entropy of the label distribution."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy
   
    def _information_gain(self, X_column, y, split_value):
        """Calculate the information gain of a split."""
        parent_entropy = self._entropy(y)
       
        left_mask = X_column <= split_value
        right_mask = X_column > split_value
       
        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])
       
        left_weight = np.sum(left_mask) / len(y)
        right_weight = np.sum(right_mask) / len(y)
       
        child_entropy = (left_weight * left_entropy + right_weight * right_entropy)
        gain = parent_entropy - child_entropy
       
        return gain
   
    def _best_split(self, X, y):
        """Find the best split for a dataset."""
        best_gain = 0
        best_split = None
        best_feature = None
       
        n_features = X.shape[1]
       
        for feature in range(n_features):
            values = np.unique(X[:, feature])
            for split_value in values:
                gain = self._information_gain(X[:, feature], y, split_value)
                if gain > best_gain:
                    best_gain = gain
                    best_split = split_value
                    best_feature = feature
       
        return best_feature, best_split
   
    def _build_tree(self, X, y):
        """Build the decision tree recursively."""
        if len(set(y)) == 1:
            return y[0]
       
        if X.shape[1] == 0:
            return Counter(y).most_common(1)[0][0]
       
        feature, split_value = self._best_split(X, y)
       
        if feature is None:
            return Counter(y).most_common(1)[0][0]
       
        left_mask = X[:, feature] <= split_value
        right_mask = X[:, feature] > split_value
       
        left_tree = self._build_tree(X[left_mask], y[left_mask])
        right_tree = self._build_tree(X[right_mask], y[right_mask])
       
        return (feature, split_value, left_tree, right_tree)
   
    def _predict_sample(self, tree, sample):
        """Predict the class of a single sample."""
        if not isinstance(tree, tuple):
            return tree
       
        feature, split_value, left_tree, right_tree = tree
       
        if sample[feature] <= split_value:
            return self._predict_sample(left_tree, sample)
        else:
            return self._predict_sample(right_tree, sample)
   
    def _plot_tree(self, ax, tree, feature_names, pos=None, x=0, y=0, dx=1.5, dy=0.8):
        """Plot the decision tree using matplotlib with more spacing."""
        if pos is None:
            pos = {}
       
        if not isinstance(tree, tuple):
            ax.text(x, y, f'Leaf: {self.class_names[tree]}', bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.5'),
                    ha='center', va='center', fontsize=8)
            pos[(x, y)] = f'Leaf: {self.class_names[tree]}'
            return pos
       
        feature, split_value, left_tree, right_tree = tree
        feature_name = feature_names[feature]
        node_label = f'{feature_name} <= {split_value}'
       
        ax.text(x, y, node_label, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'),
                ha='center', va='center', fontsize=8)
        pos[(x, y)] = node_label
       
        if isinstance(left_tree, tuple) or not isinstance(left_tree, tuple):
            pos = self._plot_tree(ax, left_tree, feature_names, pos, x - dx, y - dy, dx / 1.5, dy)
            ax.plot([x, x - dx], [y, y - dy], 'k-')
       
        if isinstance(right_tree, tuple) or not isinstance(right_tree, tuple):
            pos = self._plot_tree(ax, right_tree, feature_names, pos, x + dx, y - dy, dx / 1.5, dy)
            ax.plot([x, x + dx], [y, y - dy], 'k-')
       
        return pos

    def plot_tree(self, feature_names):
        """Create and display the graphical representation of the decision tree."""
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis('off')
        pos = self._plot_tree(ax, self.tree, feature_names)
       
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - 1, xlim[1] + 1)
        ax.set_ylim(ylim[0] - 1, ylim[1] + 1)
       
        plt.show()

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names  

id3 = ID3()
id3.fit(X, y, class_names) 

predictions = id3.predict(X)  

accuracy = np.mean(predictions == class_names[y])  
print(f"Accuracy: {accuracy:.2f}")

id3.plot_tree(iris.feature_names)

