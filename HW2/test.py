from matplotlib import pyplot as plt
import numpy as np

def print_heat_map(Cs, Sigmas, accuracies):
    fig, ax = plt.subplots()
    im = ax.imshow(accuracies)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(Sigmas)))
    ax.set_yticks(np.arange(len(Cs)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(Sigmas)
    ax.set_yticklabels(Cs)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for j in range(len(Sigmas)):
        for i in range(len(Cs)):
            text = ax.text(j, i, accuracies[i][j],
                        ha="center", va="center", color="w")

    ax.set_title("Accuracies over c and sigma")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    sigma = [0.001, 0.01, 0.1]
    accuracies = [[0.50125, 0.50125, 0.50125], [0.50125, 0.50125, 0.50125], [0.50125, 0.50125, 0.50125], [0.50125, 0.50125, 0.62875], [0.50625, 0.648125, 0.966875], [0.47625, 0.653125, 0.97375], [0.47625, 0.653125, 0.97], [0.47625, 0.653125, 0.97]]
    print_heat_map(C, sigma, accuracies)