import numpy as np
from scipy import stats
import utils

class DecisionStump:
    def __init__(self):
        pass

    def fit(self, X, y):
        # This time we don't want to discretize the data
        # We want to split the data if it is above or below a certain threshold

        #Grab shape of input array for fitting
        N, D = X.shape


        #grab mode of classification attributes
        yArrayMode = utils.mode(y)

        #prepare split variables for error honing
        splitSat = yArrayMode
        splitVariable = None
        splitValue = None
        splitNot = None

        minError = np.sum(y != yArrayMode)

        # Check if labels are not all equal
        if np.unique(y).size > 1:
            # Loop over features looking for the best split

            for d in range(D):
                for n in range(N):
                    # Choose value to equate to
                    value = X[n, d]

                    # Find most likely class for each split
                    y_sat = utils.mode(y[X[:,d] > value])
                    yNot = utils.mode(y[X[:,d] <= value])

                    # Make predictions
                    y_pred = y_sat * np.ones(N)
                    y_pred[X[:, d] <= value] = yNot

                    # Compute error
                    errors = np.sum(y_pred != y)

                    # Compare to minimum error so far
                    if errors < minError:
                        # This is the lowest error, store this value
                        minError = errors
                        splitVariable = d
                        splitValue = value
                        splitSat = y_sat
                        splitNot = yNot

        #set lowest error split variables
        self.splitVariable = splitVariable
        self.splitValue = splitValue
        self.splitSat = splitSat
        self.splitNot = splitNot

    def predict(self, X):
        M, D = X.shape

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] > self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat

