import numpy as np
import cvxopt
import pandas as pd
from matplotlib import pyplot as plt

def print_confusion_matrix(confusion_matrix, acc):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(10):
        for j in range(10):
            text = ax.text(i, j, confusion_matrix[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix, Accuracy "+str(acc))
    fig.tight_layout()
    plt.show()

def rbf_kernel(X1, X2, sigma):
    distance = np.linalg.norm(X1 - X2) ** 2
    return np.exp(-1*distance/(2*(sigma**2)))

def mfeat_svm_train(X, y, c, sigma):
    n = X.shape[0]
    
    # compute gamma matrix
    gamma_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gamma_matrix[i, j] = rbf_kernel(X[i], X[j], sigma)

    # compute optimizers parameters
    P = cvxopt.matrix(np.outer(y, y) * gamma_matrix)
    q = cvxopt.matrix(np.ones(n) * -1)
    A = cvxopt.matrix(y, (1, n),  tc='d')
    b = cvxopt.matrix(0, tc='d')
    G_max = np.identity(n) * -1
    G_min = np.identity(n)
    G = cvxopt.matrix(np.vstack((G_max, G_min)))
    h_max = cvxopt.matrix(np.zeros(n))
    h_min = cvxopt.matrix(np.ones(n) * c)
    h = cvxopt.matrix(np.vstack((h_max, h_min)))

    # compute lagrange multipliers
    minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
    lambda_ = np.ravel(minimization['x'])
    
    # compute non zero multipliers
    non_zero_indices = lambda_ > 1e-7
    lagr_multipliers = lambda_[non_zero_indices]
    
    # compute support vector
    x_support_vector = X[non_zero_indices]

    # Get the corresponding labels
    y_support_vector = y[non_zero_indices]

    # compute weights
    weights = lagr_multipliers[0] * y_support_vector[0] * y_support_vector[0]
    for i in range(1, len(lagr_multipliers)):
        weights += lagr_multipliers[i] * y_support_vector[i] * x_support_vector[i]

    # compute intercept
    intercept = y_support_vector[0]
    for i in range(len(lagr_multipliers)):
        intercept -= lagr_multipliers[i] * y_support_vector[i] * rbf_kernel(x_support_vector[i], x_support_vector[0], sigma)

    return lagr_multipliers, x_support_vector, y_support_vector, intercept

def mfeat_svm_predict(X, weights, sigma):
    y_pred = np.zeros(X.shape[0])
    pos = 0
    for s in range(X.shape[0]):
        max_h = 0
        for class_ in range(10):
            prediction = 0
            for i in range(len(weights[0][class_])):
                prediction += weights[0][class_][i] * weights[2][class_][
                    i] * rbf_kernel(weights[1][class_][i], X[s], sigma)
            prediction += weights[3][class_]
            if max_h < prediction:
                pos += 1
                max_h = prediction
                y_pred[s] = class_

    print(pos)
    return y_pred

def cross_validate():
    train_data = pd.read_csv("data\\mfeat_train.csv")

    X = train_data[train_data.columns[0:65]].to_numpy()
    y = train_data[train_data.columns[65]].to_numpy()
    y = y-1
    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    sigma = [0.1, 1, 10, 100]

    accuracies = {}

    for c in C:
        accuracies[c] = {}
        for s in sigma:
            weights = []
            intercepts = []

            alphas = []
            support_vectors = []
            support_labels = []

            for class_ in range(10):
                train_y = y.copy()
                for i in range(X.shape[0]):
                    if y[i] != class_:
                        train_y[i] = -1.0
                    else:
                        train_y[i] = 1.0

                lagr_multipliers, x_support_vector, y_support_vector, intercept = mfeat_svm_train(X, train_y, c, s)
                alphas.append(lagr_multipliers)
                support_vectors.append(x_support_vector)
                support_labels.append(y_support_vector)
                intercepts.append(intercept)
            
            acc = test([alphas, support_vectors, support_labels, intercepts], s)
            accuracies[c][s] = acc

    for c in C:
        for s in sigma:
            print("c, "+str(c)+", sigma, "+str(s)+", acc  "+str(accuracies[c][s]))


def test(weights, sigma):
    test_data = pd.read_csv("data\\mfeat_test.csv")
    X = test_data[test_data.columns[0:65]].to_numpy()
    y = test_data[test_data.columns[65]].to_numpy()

    y = y-1
        
    n = 0
    d = 0
    confusion_matrix = np.zeros((10, 10))
    total_real_count = np.zeros((10))

    y_preds = mfeat_svm_predict(X, weights, sigma)
     # compute confusion matrix
    for i in range(y_preds.shape[0]):
        total_real_count[int(y[i])] += 1
        confusion_matrix[int(y_preds[i])][int(y[i])] += 1
    
    # compute percentage
    for i in range(10):
        confusion_matrix[i][:] /= total_real_count[i]
        confusion_matrix[i][:] *= 100

    n = np.sum(y == y_preds)
    d = X.shape[0]
        
    acc = (n/d)*100
    print_confusion_matrix(confusion_matrix.astype(int), acc)
    # print("Accuracy "+str(acc))
    return acc

def train_best_model(c, sigma):
    train_data = pd.read_csv("data\\mfeat_train.csv")

    X = train_data[train_data.columns[0:65]].to_numpy()
    y = train_data[train_data.columns[65]].to_numpy()

    y = y - 1

    intercepts = []

    alphas = []
    support_vectors = []
    support_labels = []

    for class_ in range(10):
        train_y = y.copy()
        for i in range(X.shape[0]):
            if y[i] != class_:
                train_y[i] = -1.0
            else:
                train_y[i] = 1.0

        lagr_multipliers, x_support_vector, y_support_vector, intercept = mfeat_svm_train(X, train_y, c, sigma)
        alphas.append(lagr_multipliers)
        support_vectors.append(x_support_vector)
        support_labels.append(y_support_vector)
        intercepts.append(intercept)
    
    weights = [alphas, support_vectors, support_labels, intercepts]
    np.save("multi_svm_w", weights, allow_pickle=True)
    # print(test(weights, sigma))

def test_saved_model(sigma):
    weights = np.load("multi_svm_w.npy", allow_pickle=True)
    print(test(weights, sigma))

if __name__ == "__main__":
    # cross_validate()
    # obtained following values after running above function
    c = 1
    sigma = 100

    # train_best_model(c, sigma)
    test_saved_model(sigma)