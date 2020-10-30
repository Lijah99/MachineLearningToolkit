import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy import stats


#This file holds utility functions that are not specific to any classifier
#Created by Elijah Flinders

#plots the decision regions given X, y, and a classifier
def plot_decision_regions(X, y, classifier, title="Graph", resolution = 0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
               np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, s = 15,
                    c = cmap(idx), marker = markers[idx], label = cl)

    #show plot
    plt.title(title)
    plt.show()

def mode(y):
        #Computes the element with the maximum count
        #y : an input numpy array
        #Returns the element with the maximum count

        if len(y)==0:
            return -1
        else:
            return stats.mode(y.flatten())[0][0]