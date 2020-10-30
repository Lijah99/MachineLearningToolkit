from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

#Perceptron class, functions as a rudimentary basis for neural nets
#Created by Elijah Flinders, 2020
class Perceptron(object):

    #class initialization
    def __init__(self, learning_rate = 0.01, no_iter = 10):
        self.learning_rate = learning_rate
        self.no_iter = no_iter

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    #function to determine the fit
    #X = Set of training vectors, shape = [#samples, #features]
    #Y = Set of target values, shape = [#samples]
    def fit(self, X, Y):

        # weights: create a weights array of right size
        # and initialize elements to zero
        
        self.weights = np.zeros(X.shape[1] + 1)

        # Number of misclassifications, creates an array
        # to hold the number of misclassifications
        self.misclass = np.array([])

        # main loop to fit the data to the labels
        for i in range(self.no_iter):
            # set iteration error to zero
            errors = 0
            #loop over all the objects in X and corresponding y element
            for xi, target in zip(X, Y):
                #calculate the needed (deltaW) update from previous step
                deltaW = self.learning_rate * (target - self.predict(xi))
                #calculate what the current object will add to the weight
                self.weights[1:] += deltaW * xi
                #set the bias to be the current deltaW
                self.weights[0] += deltaW
                #if deltaW not 0, increase error iter
                if deltaW != 0:
                    errors += 1

            # If data converged, exit early
            if(errors == 0):
                self.misclass = np.append(self.misclass, errors)
                return self

            # Update the misclassification array with # of errors in iteration
            self.misclass = np.append(self.misclass, errors)

        return self

    #will determine the net inputs
    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

#plots the decision regions given X, y, and a classifier
def plot_decision_regions(X, y, classifier, resolution = 0.02):
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
    plt.show()