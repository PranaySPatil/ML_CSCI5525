import numpy as np
import matplotlib.pyplot as plt
iterations = 500

def lsq(a, b):
    a_dagger = np.linalg.pinv(a)
    w_hat = np.dot(a_dagger, b)
    return w_hat

def lsq_iter(a, b):
    w_hat = lsq(a, b)
    w = np.zeros((iterations, w_hat.shape[0], w_hat.shape[1]))
    diff = np.zeros(iterations)
    diff[0] = np.linalg.norm(w[0]-w_hat, 1)
    err = np.zeros(iterations)
    err[0] = np.linalg.norm(np.dot(a, w[0])-b, 2)
    mu = 1/np.linalg.norm(a)

    for i in range(1, iterations):
        w[i] = w[i-1] - mu * np.dot(a.T, np.dot(a, w[i-1]) - b)
        diff[i] = np.linalg.norm(w[i]-w_hat, 1)
        err[i] = np.linalg.norm(np.dot(a, w[i])-b, 2)

    plt.plot(list(range(0, iterations)), diff)
    plt.xlabel('k')
    plt.ylabel('||w(k)-w||')
    plt.show()

    plt.plot(list(range(0, iterations)), err)
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.show()

if __name__ == "__main__":
    a = np.random.normal(0, 0.1, (20, 10))
    b = np.random.normal(0, 0.1, (20, 1))

    lsq_iter(a, b)