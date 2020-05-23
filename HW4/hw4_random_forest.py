import pandas as pd
from sklearn import tree
import numpy as np
from matplotlib import pyplot as plt

# computed by running algo over all the features, and sorting by featre_importance
sorted_features_by_importance = [221,  76, 154,  15,  86, 197, 227,  92, 246, 209,  20,  10, 202,
        70, 234, 196, 239, 147, 226,  32, 175,  21, 140,   2, 156, 133,
         5,  41, 113, 134, 123,  82, 137, 158,  89, 119, 165,  38,  87,
        46, 129,  72,  13, 225, 163,  22, 112,  78, 249,  43, 214,  12,
       233, 238, 243, 126,  42, 135,  23, 167, 108, 118, 172, 125, 104,
        81, 120,  84, 241, 127, 159,  68, 168,  53,  75, 155, 182, 208,
       244, 174, 150, 215,  44,  83, 216,  19,  96, 141,   9, 107,  93,
       232,  69, 128,  36, 161, 245, 178,  55,  28,  59, 207, 109,  62,
       142, 195, 247, 117, 173, 200,  56,  54,  66, 146,  74, 193, 115,
        45, 189,  34,  37, 217, 213, 240,   1, 224, 235, 179,  67, 151,
        39, 152, 176,  30,  51,  50, 184,  58,  61, 139,  65, 231, 242,
        17,  29, 228, 148, 132, 191, 192,  95, 121,  48,  97, 105, 183,
        27, 116,  85,  24,  40,  49, 122,  91,  16,  47,  35,  14,  18,
        52,  64,  33,  63,  31,  11,   6,  57,  26,  60,   3,   4,   8,
        25,   7, 124,  71, 169, 171, 177, 180, 181, 185, 186, 187, 188,
       190, 194, 198, 199, 201, 203, 204, 205, 206, 210, 211, 212, 218,
       219, 220, 222, 223, 229, 230, 236, 237, 170, 166,  73, 164,  77,
        79,  80,  88,  90,  94,  98,  99, 100, 101, 102, 103, 106, 110,
       111, 114, 248, 130, 131, 136, 138, 143, 144, 145, 149, 153, 157,
       160, 162,   0]

def test_random_forest(decision_trees, X, y):
    y_ = np.zeros(y.shape[0])

    for tree, feature_idx in decision_trees:
        y_ += tree.predict(X[:, feature_idx])
    
    y_ = [0 if a/len(decision_trees)<0.5 else 1 for a in y_]
    acc = np.sum(y == y_)/y.shape[0]
    err = 1-acc
    return acc   

def train_random_forest(n_trees, n_features):
    train_data = pd.read_csv("health_train.csv")
    X = train_data[train_data.columns[0:-1]].to_numpy()
    y = train_data[train_data.columns[-1]].to_numpy()

    classifier = tree.DecisionTreeClassifier(criterion='gini')
    decision_trees = []

    for t in range(n_trees):
        # sample data points with replacement
        shuffled_idx = np.random.choice(range(X.shape[0]), y.shape[0], replace=True)
        shuffled_X = X[shuffled_idx]
        shuffled_y = y[shuffled_idx]

        # select most important 'n_features'
        feature_idx = sorted_features_by_importance[0:n_features]

        # train decision tree
        decision_trees.append((classifier.fit(shuffled_X[:,feature_idx], shuffled_y), feature_idx))
    
    # test random forest
    train_acc = test_random_forest(decision_trees, X, y)

    test_data = pd.read_csv("health_test.csv")
    X = test_data[train_data.columns[0:-1]].to_numpy()
    y = test_data[train_data.columns[-1]].to_numpy()
    test_acc = test_random_forest(decision_trees, X, y)

    return train_acc, test_acc

def train_over_tree_size():
    # initialize
    tree_sizes = [10, 20, 40, 80, 100]
    train_accs = []
    test_accs = []

    for s in tree_sizes:
        # train forest
        train_acc, test_acc = train_random_forest(s, 250)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    plt.plot(tree_sizes, train_accs, label='Training Acc')
    plt.plot(tree_sizes, test_accs, label='Testing Acc')
    plt.xlabel('#Trees')
    plt.ylabel('Accuracies')
    plt.title("Accuracies vs #Trees")
    plt.legend()
    plt.show()

def train_over_feature_sets():
    # initialize
    feature_sizes = [50, 100, 150, 200, 250]
    train_accs = []
    test_accs = []

    for s in feature_sizes:
        # train forest
        train_acc, test_acc = train_random_forest(100, s)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    plt.plot(feature_sizes, train_accs, label='Training Acc')
    plt.plot(feature_sizes, test_accs, label='Testing Acc')
    plt.xlabel('#Features')
    plt.ylabel('Accuracies')
    plt.title("Accuracies vs #Features")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # train_random_forest(100, 250)
    # train_over_tree_size()
    train_over_feature_sets()