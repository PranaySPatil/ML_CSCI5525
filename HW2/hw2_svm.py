import numpy as np
import cvxopt
import pandas as pd
from matplotlib import pyplot as plt

def shuffle_data_in_k_folds(X, y, k):
    # shuffles x and y with same permutation
    p = np.random.permutation(X.shape[0])
    X_shuffled = X[p]
    y_shuffled = y[p]

    return np.array(np.array_split(X_shuffled, k)), np.array(np.array_split(y_shuffled, k))

def  get_next_train_valid(X_shuffled, y_shuffled, iter):
    X_val, y_val = X_shuffled[iter-1], y_shuffled[iter-1]
    X_train = []
    y_train = []
    for i in range(X_shuffled.shape[0]):
        if i != iter-1:
            X_train.extend(X_shuffled[i])
            y_train.extend(y_shuffled[i])

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

def svmfit(X, y, c):
    n = X.shape[0]

    # compute gamma matrix
    gamma_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gamma_matrix[i, j] = np.inner(X[i], X[j])

    # compute optimizers parameters
    P = cvxopt.matrix(np.outer(y, y) * gamma_matrix)
    q = cvxopt.matrix(np.ones(n) * -1)
    A = cvxopt.matrix(y, (1, n))
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

    # compute intercept
    intercept = y_support_vector[0]
    for i in range(len(lagr_multipliers)):
        intercept -= lagr_multipliers[i] * y_support_vector[i] * np.inner(x_support_vector[i], x_support_vector[0])

    weights = {
        "alpha": lagr_multipliers,
        "x": x_support_vector,
        "y": y_support_vector,
        "intercept": intercept
    }

    return weights

def predict(X, weights):
    y_pred = []
    
    for sample in X:
        prediction = 0
        for i in range(len(weights["alpha"])):
            prediction += weights["alpha"][i] * weights["y"][i]* np.dot(weights["x"][i], sample)
        prediction += weights["intercept"]
        y_pred.append(np.sign(prediction))
    return np.array(y_pred)

def k_fold_cv(train_data, test_data, k, c):
    accuracies = []
    train_accuracy = 0
    cv_accuracy = 0
    test_accuracy = 0

    for iter in range(0, k):
        # train
        X_train, y_train, X_valid, y_valid = get_next_train_valid(train_data['X'], train_data['y'], iter+1)
        weights = svmfit(X_train, y_train, c)

        # compute training accuracy
        y_labels = predict(X_train, weights)
        n = np.sum(y_train == y_labels)
        d = X_train.shape[0]
        train_accuracy += (n/d)

        # compute validation accuracy
        y_labels = predict(X_valid, weights)
        n = np.sum(y_valid == y_labels)
        d = X_valid.shape[0]
        cv_accuracy += (n/d)

        # compute test accuracy
        y_labels = predict(test_data['X'], weights)
        n = np.sum(test_data['y'] == y_labels)
        d = test_data['X'].shape[0]
        test_accuracy += (n/d)

    return train_accuracy/10, cv_accuracy/10, test_accuracy/10

def plot_accuracies_over_c(accuracies, c):
    plt.plot(c, accuracies[:, 0], label='Training Acc')
    plt.plot(c, accuracies[:, 1], label='Validation Acc')
    plt.plot(c, accuracies[:, 2], label='Test Acc')
    plt.xlabel('C')
    plt.ylabel('Accuracies')
    # plt.axis(c)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data\\hw2data.csv",  header=None)
    train_df = df.sample(frac=0.8, random_state=200)
    test_df = df.drop(train_df.index)
    X_train = train_df[train_df.columns[0:2]].to_numpy()
    y_train = train_df[train_df.columns[2]].to_numpy()
    X_test = test_df[test_df.columns[0:2]].to_numpy()
    y_test = test_df[test_df.columns[2]].to_numpy()
    k = 10
    X_shuffled, y_shuffled = shuffle_data_in_k_folds(X_train, y_train, k)
    train_data = {'X': X_shuffled, 'y': y_shuffled}
    test_data = {'X': X_test, 'y': y_test}

    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    accuracies = []
    for c in C:
        train_accuracy, cv_accuracy, test_accuracy = k_fold_cv(train_data, test_data, k, c)
        accuracies.append([train_accuracy, cv_accuracy, test_accuracy])
    
    plot_accuracies_over_c(np.array(accuracies), C)
    print(accuracies)