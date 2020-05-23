import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt

def get_margin(X, y, trees, alphas):
    margin = np.zeros(y.shape[0])

    # compute margin
    for i in range(y.shape[0]):
        for t in range(len(trees)):
            margin[i] += alphas[t]*trees[t].predict(np.reshape(X[i], (1,-1)))
    
    # compute margin
    margin = margin/np.sum(alphas)
    margin *= y

    return margin

def get_misclassification_error(X, y, trees, alphas):
    y_pred = np.zeros(y.shape[0])

    for t in range(len(trees)):
        y_pred += alphas[t]*trees[t].predict(X)
    
    y_pred = np.sign(y_pred)

    return 1 - (np.sum(y_pred==y)/y.shape[0])

def adaboost(train_X, train_y, test_X, test_y, T):
    # initialize
    D_t = np.array([1/train_X.shape[0]]*train_X.shape[0])
    trees = []
    alphas = []
    classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=1)
    train_errors = []
    test_errors = []

    for t in range(T):
        # compute h_t
        h_t = classifier.fit(train_X, train_y, sample_weight=D_t)
        y_pred = h_t.predict(train_X)

        # compute error
        epsilon_t = 0
        for i in range(train_y.shape[0]):
            if not train_y[i] == y_pred[i]:
                epsilon_t += D_t[i]

        # compute alpha
        alpha_t = 0.5*np.log((1-epsilon_t)/(epsilon_t+np.finfo(float).eps))

        # compute sample weights for next tree
        D_t = D_t*np.exp(alpha_t*train_y*y_pred)
        D_t = D_t/np.sum(D_t)

        alphas.append(alpha_t)
        trees.append(h_t)

        # compute train and test errors
        train_errors.append(get_misclassification_error(train_X, train_y, trees, alphas))
        test_errors.append(get_misclassification_error(test_X, test_y, trees, alphas))

    plt.plot(range(T+1)[1:], train_errors, label="Training Error")
    plt.plot(range(T+1)[1:], test_errors, label="Test Error")
    plt.title("Error  vs #Trees")
    plt.xlabel('#Trees')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    return trees, alphas

def train_adaboost():
    train_data = pd.read_csv("cancer_train.csv")
    train_X = train_data[train_data.columns[0:-1]].to_numpy()
    train_y = train_data[train_data.columns[-1]].to_numpy()
    train_y[train_y==0] = -1
    test_data = pd.read_csv("cancer_test.csv")
    test_X = test_data[test_data.columns[0:-1]].to_numpy()
    test_y = test_data[test_data.columns[-1]].to_numpy()
    test_y[test_y==0] = -1

    # train trees
    trees, alphas = adaboost(train_X, train_y, test_X, test_y, 100)

def compute_margin_distribution():
    train_data = pd.read_csv("cancer_train.csv")
    train_X = train_data[train_data.columns[0:-1]].to_numpy()
    train_y = train_data[train_data.columns[-1]].to_numpy()
    train_y[train_y==0] = -1
    test_data = pd.read_csv("cancer_test.csv")
    test_X = test_data[test_data.columns[0:-1]].to_numpy()
    test_y = test_data[test_data.columns[-1]].to_numpy()
    test_y[test_y==0] = -1

    tree_cnt = [25, 50, 75,100]
    for t in tree_cnt:
        # train trees
        trees, alphas = adaboost(train_X, train_y, test_X, test_y, t)

        # compute margin
        margin = get_margin(train_X, train_y, trees, alphas)

        # compute cummulitive sum and plot the graph
        counts, bin_edges = np.histogram(margin)
        cdf = np.cumsum (counts)
        plt.plot(bin_edges[1:], cdf/cdf[-1])
        plt.title("Trees # "+str(t))
        plt.show()

if __name__ == "__main__":
    train_adaboost()
    # compute_margin_distribution()