import test_error_rate
import numpy as np
from matplotlib import pyplot as plt

N = 21283

def get_weak_learner(errors):
    min_ht = np.random.randint(0, 50, N)
    min_score = N
    if len(errors)>0:
        threshold = errors[-1]
    else:
        threshold = 0.02

    for i in range(100):
        h_t = np.random.randint(0, 50, N)
        error = test_error_rate.error_rate.score(h_t)/N
        print(error)
        if min_score > error:
            min_ht = h_t
            min_score = error
        if error < threshold:
            min_ht = h_t
            min_score = error
            print("Found min h_t")
            break

    
    return min_ht, min_score

def compute_error_for_i_models(h, alpha):
    predictions = [0]*N
    for i in range(len(h)):
        predictions += (alpha[i]*h[i])
        # predictions -= predictions.min()
        # predictions = predictions/predictions.max()

    predictions = [1 if y>0.5 else 0 for y in predictions]

    return test_error_rate.error_rate(np.array(predictions))


def boosting(t):
    h = []
    alpha = [1]
    err = []
    h_t = np.random.uniform(0, 1, N)
    h.append(h_t)
    # err.append(test_score.score(h_t))

    for i in range(t):
        error_t = compute_error_for_i_models(h, alpha)
        
        err.append(error_t)
        alpha_t = 0.5*np.log((1-(error_t/N))/(error_t/N))
        alpha.append(alpha_t)
        gradient = np.random.multinomial(error_t, np.ones(N)/N, size=1)[0]
        # gradient = np.random.uniform(0,1,N)
        h.append(gradient)
        # errors.append(compute_error_for_i_models(h, alpha))
        # gradient2 = alpha_t*gradient
        # h_t += gradient2.astype(int)
        # h_t -= h_t.min()
        # h_t = h_t/h_t.max()
        # h_t *= 50
        print(error_t)

    
    # plt.plot(errors)
    # plt.title("Error  vs #Trees")
    # plt.ylabel("Error")
    # plt.xlabel("#Trees")
    # plt.show()

    plt.plot(err)
    plt.title("Error  vs #Trees")
    plt.ylabel("Error")
    plt.xlabel("#Trees")
    plt.show()

        

if __name__ == "__main__":
    boosting(100)
