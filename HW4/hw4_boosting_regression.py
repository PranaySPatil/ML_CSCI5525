import test_score
import numpy as np
from matplotlib import pyplot as plt

N = 21283

def compute_error_for_i_models(h, alpha):
    predictions = [0]*N

    for i in range(len(h)):
        # add weighted outputs of each model
        predictions += (alpha[i]*h[i]).astype(int)

        # normalize it to the range [0, 50]
        predictions -= predictions.min()
        predictions = predictions/predictions.max()
        predictions *= 50

    predictions = [y for y in predictions]

    return test_score.score(np.array(predictions))

def boosting(t):
    h = []
    alpha = [1]
    err = []
    # initialize first model, we will try to minimize error from this model's error
    h_t = np.random.randint(0, 50, N)
    h.append(h_t)

    # compute gradient and try mimizing the error per iteration
    for i in range(t):
        # error computation
        error_t = compute_error_for_i_models(h, alpha)
        err.append(error_t)

        # compute alpha given the error
        alpha_t = 0.5*np.log((1-(error_t/N))/(error_t/N))
        alpha.append(alpha_t)

        # guess the gradient vector whose sum is bounded by error
        gradient = np.random.multinomial(int(error_t), np.ones(N)/N, size=1)[0]
        
        # add the gradient to model
        h.append(gradient)
        print(error_t)

    plt.plot(err)
    plt.title("Error  vs #Trees")
    plt.ylabel("Error")
    plt.xlabel("#Trees")
    plt.show()

if __name__ == "__main__":
    boosting(100)
