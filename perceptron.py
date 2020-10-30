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