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
    
def rbf_kernel(X1, X2, sigma):
    distance = np.linalg.norm(X1 - X2) ** 2
    return np.exp(-1*distance/(2*(sigma**2)))

def rbf_svmfit(X, y, c, sigma):
    n = X.shape[0]
    gamma_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gamma_matrix[i, j] = rbf_kernel(X[i], X[j], sigma)

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

    minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
    lagr_mult = np.ravel(minimization['x'])
    idx = lagr_mult > 1e-7
    # Get the corresponding lagr. multipliers
    lagr_multipliers = lagr_mult[idx]
    # Get the samples that will act as support vectors
    support_vectors = X[idx]
    # Get the corresponding labels
    support_vector_labels = y[idx]

    # Calculate intercept with first support vector
    intercept = support_vector_labels[0]
    for i in range(len(lagr_multipliers)):
        intercept -= lagr_multipliers[i] * support_vector_labels[
            i] * rbf_kernel(support_vectors[i], support_vectors[0], sigma)

    return lagr_multipliers, support_vectors, support_vector_labels, intercept

def predict(X, lagr_multipliers, support_vectors, support_vector_labels, intercept, sigma):
    y_pred = []
    
    for sample in X:
        prediction = 0
        for i in range(len(lagr_multipliers)):
            prediction += lagr_multipliers[i] * support_vector_labels[
                i] * rbf_kernel(support_vectors[i], sample, sigma)
        prediction += intercept
        y_pred.append(np.sign(prediction))
    return np.array(y_pred)

def print_heat_map(Cs, Sigmas, accuracies, title):
    fig, ax = plt.subplots()
    im = ax.imshow(accuracies)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(Sigmas)))
    ax.set_yticks(np.arange(len(Cs)))
    # ... and label them with the respective list entries
    ax.set_yticklabels(Cs)
    ax.set_xticklabels(Sigmas)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(Sigmas)):
        for j in range(len(Cs)):
            text = ax.text(i, j, accuracies[j][i],
                        ha="center", va="center", color="w")

    ax.set_title(title+" over c and sigma")
    fig.tight_layout()
    plt.show()

def cross_validate():
    df = pd.read_csv("data\\hw2data.csv",  header=None)
    train_df = df.sample(frac=0.8, random_state=200)
    test_df = df.drop(train_df.index)
    X_train = train_df[train_df.columns[0:2]].to_numpy()
    y_train = train_df[train_df.columns[2]].to_numpy()
    X_test = test_df[test_df.columns[0:2]].to_numpy()
    y_test = test_df[test_df.columns[2]].to_numpy()
    k = 10

    X_shuffled, y_shuffled = shuffle_data_in_k_folds(X_train, y_train, k)

    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    sigma = [0.001, 0.01, 0.1]
    
    accuracies = []
    errors = []
    validation_acc = []
    for c in C:
        accuracy = []
        error = []
        for s in sigma:
            n = 0
            d = 0
            for iter in range(0, k):
                X_train, y_train, X_valid, y_valid = get_next_train_valid(X_shuffled, y_shuffled, iter+1)
                lagr_multipliers, support_vectors, support_vector_labels, intercept = rbf_svmfit(X_train, y_train, c, s)
                y_labels = predict(X_valid, lagr_multipliers, support_vectors, support_vector_labels, intercept, s)

                n += np.sum(y_valid == y_labels)
                d += X_valid.shape[0]

            accuracy.append(n/d)
            error.append(1-(n/d))
        accuracies.append(accuracy)
        errors.append(error)
    
    print(accuracies)

    print_heat_map(C, sigma, accuracies, "Accuracy")
    print_heat_map(C, sigma, errors, "Error")

def train_best_model(c, sigma):
    df = pd.read_csv("data\\hw2data.csv",  header=None)
    train_df = df.sample(frac=0.8, random_state=200)
    test_df = df.drop(train_df.index)
    X_train = train_df[train_df.columns[0:2]].to_numpy()
    y_train = train_df[train_df.columns[2]].to_numpy()
    X_test = test_df[test_df.columns[0:2]].to_numpy()
    y_test = test_df[test_df.columns[2]].to_numpy()
    
    lagr_multipliers, support_vectors, support_vector_labels, intercept = rbf_svmfit(X_train, y_train, c, sigma)
    y_labels = predict(X_test, lagr_multipliers, support_vectors, support_vector_labels, intercept, sigma)

    accuracy = np.sum(y_test == y_labels)/X_test.shape[0]
    print("Accuracy: "+str(accuracy))

if __name__ == "__main__":
    # cross_validate()
    train_best_model(100, 0.1)