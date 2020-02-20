import numpy as np
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
    for i in range(X_shuffled.size):
        if i != iter-1:
            X_train.extend(X_shuffled[i])
            y_train.extend(y_shuffled[i])

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

# adding X_valid and y_valid to plot their error rates for each iteration
def train(X_train, y_train, X_valid=None, y_valid=None):
    # initializing params
    w = np.random.uniform(0, 1, (X_train.shape[1], y_train.shape[1]))
    b = np.random.uniform(0, 1, (1, 1))[0][0]
    learning_rate = 0.1
    training_error_rates = []
    validation_error_rates = []

    for i in range(5000):
        gradient_w = 0
        gradient_b = 0
        for i in range(X_train.shape[0]):
            # computing prediction
            y_pred = 1 / (1 + np.exp(-1 * ( np.dot(X_train[i], w) + b)))
            # computing gradients across all samples
            gradient_w += np.dot(X_train[i].reshape(1, X_train.shape[1]).T,  (y_pred - y_train[i]).reshape(1,1))
            gradient_b += (y_train[i]-y_pred)[0]
        
        # updating params
        w -= learning_rate*(gradient_w/X_train.shape[0])
        b -= learning_rate*(gradient_b/X_train.shape[0])

        # calculating training error for this iteration
        y_train_predict_class = predict(X_train, w, b)
        true_positive_class1, false_positive_class1, true_positive_class2, false_positive_class2 = compute_accuracy(y_train_predict_class, y_train)
        training_error_rates.append((false_positive_class1 + false_positive_class2)/len(y_train))

        # calculating validation error for this iteration
        y_valid_predict_class = predict(X_valid, w, b)
        true_positive_class1, false_positive_class1, true_positive_class2, false_positive_class2 = compute_accuracy(y_valid_predict_class, y_valid)
        validation_error_rates.append((false_positive_class1 + false_positive_class2)/len(y_valid))

        # stopping the algorithm if the update to w is very small
        if (np.linalg.norm(gradient_w/X_train.shape[0]) < 0.001):
            break
    
    plt.plot(list(range(0, len(training_error_rates))), training_error_rates, label='Trainin Error')
    plt.plot(list(range(0, len(validation_error_rates))), validation_error_rates, label='Validation Error')
    plt.xlabel('#iterations')
    plt.ylabel('error')
    plt.legend()
    plt.show()
    return w, b

def predict(X_valid, model_weights, model_intercept):
    y_predict_class = np.zeros((X_valid.shape[0],1))
    for i in range(X_valid.shape[0]):
        y_predict_class[i] = np.round(1 / (1 + np.exp(-1 * ( np.dot(X_valid[i], model_weights) + model_intercept)))).astype(int)
    
    return y_predict_class

def compute_accuracy(y_pred, y):
    true_positive_class1 = 0
    false_positive_class1 = 0
    true_positive_class2 = 0
    false_positive_class2 = 0
    accuracy = 0
    for i in range(y.shape[0]):
        if y[i] == 0 and y_pred[i] == 0:
            true_positive_class1 += 1
            accuracy += 1
        if y[i] == 1 and y_pred[i] == 1:
            true_positive_class2 += 1
            accuracy += 1
        if y[i] == 1 and y_pred[i] == 0:
            false_positive_class1 += 1
        if y[i] == 0 and y_pred[i] == 1:
            false_positive_class2 += 1

    return true_positive_class1, false_positive_class1, true_positive_class2, false_positive_class2

def plot_confusion_matrix(metrics, iter):
    true_positive_class1, false_positive_class1, true_positive_class2, false_positive_class2 = metrics
    fig = plt.figure(dpi=80)
    ax = fig.add_subplot(1,1,1)
    table_data=[
        ["Classes/Prediction", 0, 1],
        [0, true_positive_class1, false_positive_class1],
        [1, false_positive_class2, true_positive_class2]
    ]
    table = ax.table(cellText=table_data, loc='center')
    table.set_fontsize(14)
    table.scale(1,4)
    ax.axis('off')
    plt.title('Confusion matrix #iteration '+str(iter+1))
    plt.show()

if __name__ == "__main__":
    features_df = pd.read_csv("C:\\Users\\PranayDev\\Documents\\UMN\\ML_Assignments\\HW1\\HW1-data\\IRISFeat.csv")
    labels_df = pd.read_csv("C:\\Users\\PranayDev\\Documents\\UMN\\ML_Assignments\\HW1\\HW1-data\\IRISlabel.csv")

    X = features_df.to_numpy()
    y = labels_df.to_numpy()
    k = 5

    X_shuffled, y_shuffled = shuffle_data_in_k_folds(X, y, k)
    accuracies = {}

    for iter in range(0, k):
        X_train, y_train, X_valid, y_valid = get_next_train_valid(X_shuffled, y_shuffled, iter+1)
        model_weights, model_intercept = train(X_train, y_train, X_valid, y_valid)
        y_predict_class = predict(X_valid, model_weights, model_intercept)
        true_positive_class1, false_positive_class1, true_positive_class2, false_positive_class2 = compute_accuracy(y_predict_class, y_valid)
        accuracies[iter] = (true_positive_class1, false_positive_class1, true_positive_class2, false_positive_class2)

    for iter in accuracies:
        plot_confusion_matrix(accuracies[iter], iter)