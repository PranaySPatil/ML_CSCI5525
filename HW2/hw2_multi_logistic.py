import numpy as np
import cvxopt
import pandas as pd
from matplotlib import pyplot as plt

batch_size = 32

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

def get_mini_batch(X_train, y_train, batch_size):
    # TO DO
    mini_batch_x = []
    mini_batch_y = []
    k = 0

    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p]
    y_train = y_train[p]

    for i in range (0, X_train.shape[0], batch_size):
        batch_x = X_train[i: min(X_train.shape[0], i + batch_size), :]
        batch_label = y_train[i: min(X_train.shape[0], i + batch_size)]
        
        mini_batch_x.append(batch_x)
        mini_batch_y.append(batch_label)
        k += 1

    return np.array(mini_batch_x), np.array(mini_batch_y)

def softmax(y):
    e_power_y = np.exp(y-max(y))
    soft_max_y = e_power_y/(sum(e_power_y)+np.finfo(float).eps)

    return soft_max_y

def loss_cross_entropy_softmax(y_pred, y):
    return -1 * (y * np.log(y_pred + np.finfo(float).eps))

def mnist_predict(w, b, X):
    y_predict_class = np.zeros((X.shape[0],1))

    for i in range(X.shape[0]):
        x = X[i].reshape(X[i].shape[0], 1)
        y_pred = np.dot(w, x) + b
        y_pred = softmax(y_pred)
        y_predict_class[i] = np.argmax(y_pred)
    
    return y_predict_class

def mnist_train(mini_batch_x, mini_batch_y, learning_rate):
    mean = 0
    sigma = 5 
    w = np.random.normal(mean, sigma, (10, mini_batch_x[0].shape[1]))
    b = np.random.normal(mean, sigma, (10, 1))

    lambda_decay = 1
    training_error_rates = []
    k = 0
    batch_count = 1

    for iteration in range(1, 100000):
        print("Epoch #"+str(iteration))
        print("Batch count #"+str(k+1))
        gradient_w = np.zeros((10, mini_batch_x[0].shape[1]))
        gradient_b = np.zeros((10, 1))
        loss = 0
        if iteration % 5000 == 0:
            learning_rate = lambda_decay * learning_rate

        for i in range(mini_batch_x[k].shape[0]):
            x = mini_batch_x[k][i]
            y = mini_batch_y[k][i]
            x = x.reshape((x.shape[0], 1))
            # computing prediction
            y_pred = np.dot(w, x) + b
            y_pred = softmax(y_pred)
            
            # computing loss
            label_one_hot = np.zeros(10)
            label_one_hot[y] = 1
            loss += np.linalg.norm(loss_cross_entropy_softmax(y_pred, label_one_hot))

            # computing gradients across all samples
            gradient_w += np.dot(x,  y_pred.reshape(1, 10) - label_one_hot.reshape(1, 10)).T
            gradient_b += (y_pred.reshape(10, 1) - label_one_hot.reshape(10, 1))
        
        training_error_rates.append(loss/mini_batch_x[k].shape[0])
        # updating params
        w -= learning_rate*(gradient_w/mini_batch_x[k].shape[0])
        b -= learning_rate*(gradient_b/mini_batch_x[k].shape[0])

        # calculating training error for this iteration
        # y_train_predict_class = predict(w, b, X)
        # accuracy = np.sum(y==y_train_predict_class)/y.shape[0]

        # stopping the algorithm if the update to w is very small
        # if (np.linalg.norm(gradient_w/mini_batch_x[k].shape[0]) < 0.0000001):
        #     break

        k += 1
        k %= mini_batch_x.shape[0]
        batch_count += 1

    n = 0
    d = 0
    for i in range(mini_batch_x.shape[0]):
        y_preds = mnist_predict(w, b, mini_batch_x[i])
        n += np.sum(mini_batch_y[i].reshape(mini_batch_y[i].shape[0], 1) == y_preds)
        d += mini_batch_x[i].shape[0]
    acc = (n/d)*100

    print("Training Accuracy "+str(acc))

    return w, b

def train_mnist():
    train_data = pd.read_csv("data\\mnist_train.csv")
    test_data = pd.read_csv("data\\mnist_test.csv")

    X = train_data[train_data.columns[1:]].to_numpy()
    y = train_data[train_data.columns[0]].to_numpy()

    X = X/255

    mini_batch_x, mini_batch_y = get_mini_batch(X, y, batch_size)

    w, b = mnist_train(mini_batch_x, mini_batch_y, 0.2)
    np.save("w_mnist", w)
    np.save("b_mnist", b)

    print("Model saved")


def test_mnist():
    w = np.load("w_mnist.npy")
    b = np.load("b_mnist.npy")

    test_data = pd.read_csv("data\\mnist_test.csv")
    X = test_data[test_data.columns[1:]].to_numpy()
    y = test_data[test_data.columns[0]].to_numpy()

    n = 0
    d = 0
    y_preds = mnist_predict(w, b, X)
    y = y.reshape(y.shape[0], 1)
    confusion_matrix = np.zeros((10, 10))
    total_real_count = np.zeros((10))

    # compute confusion matrix
    for i in range(y_preds.shape[0]):
        total_real_count[int(y[i][0])] += 1
        confusion_matrix[int(y_preds[i][0])][int(y[i][0])] += 1
    
    # compute percentage
    for i in range(10):
        confusion_matrix[i][:] /= total_real_count[i]
        confusion_matrix[i][:] *= 100

    n = np.sum(y == y_preds)
    d = X.shape[0]
        
    acc = (n/d)*100
    print("Test Accuracy "+str(acc))
    print_confusion_matrix(confusion_matrix.astype(int), acc)

def train_mfeat():
    train_data = pd.read_csv("data\\mfeat_train.csv")
    test_data = pd.read_csv("data\\mfeat_test.csv")

    X = train_data[train_data.columns[0:65]].to_numpy()
    y = train_data[train_data.columns[65]].to_numpy()
    # converting range from 1-10 to 0-9
    y = y-1

    mini_batch_x, mini_batch_y = get_mini_batch(X, y, batch_size)

    w, b = mnist_train(mini_batch_x, mini_batch_y, 0.1)
    np.save("w_mfeat", w)
    np.save("b_mfeat", b)

    print("Model saved")

def test_mfeat():
    w = np.load("w_mfeat.npy")
    b = np.load("b_mfeat.npy")

    test_data = pd.read_csv("data\\mfeat_test.csv")
    X = test_data[test_data.columns[0:65]].to_numpy()
    y = test_data[test_data.columns[65]].to_numpy()
    # converting range from 1-10 to 0-9
    y = y-1

    n = 0
    d = 0
    y_preds = mnist_predict(w, b, X)
    y = y.reshape(y.shape[0], 1)
    confusion_matrix = np.zeros((10, 10))
    total_real_count = np.zeros((10))

    # compute confusion matrix
    for i in range(y_preds.shape[0]):
        total_real_count[int(y[i][0])] += 1
        confusion_matrix[int(y_preds[i][0])][int(y[i][0])] += 1
    
    # compute percentage
    for i in range(10):
        confusion_matrix[i][:] /= total_real_count[i]
        confusion_matrix[i][:] *= 100

    n = np.sum(y == y_preds)
    d = X.shape[0]
        
    acc = (n/d)*100
    print("Test Accuracy "+str(acc))
    print_confusion_matrix(confusion_matrix.astype(int), acc)

if __name__ == "__main__":
    # For mnist dataset
    # train_mnist()
    test_mnist()

    # For mfeat dataset
    # train_mfeat()
    # test_mfeat()