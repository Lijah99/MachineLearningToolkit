import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
# Support Vector Machine: Elijah Flinders
class SVM:

    # set up initialization for classifier, iterations and learning rate
    def __init__(self, iterations=10000, learningRate=0.000001):
        # set default values of learning rate and iterations
        self.iterations = iterations
        self.learningRate = learningRate

    # main fit function for the support vector machine
    def fit(self):
        # set csv data to breast cancer data
        data = pd.read_csv('breastCancerData.csv')

        #drop last column and unnecessary id column in first
        data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

        #convert categorical labels to numbers
        #M-malignant tumor, B-benign tumor
        diagnosisMap = {'M': 1.0, 'B': -1.0}
        data['diagnosis'] = data['diagnosis'].map(diagnosisMap)

        # seperate features and expected outputs on different axes
        Y = data.loc[:, 'diagnosis']
        X = data.iloc[:, 1:]

        # remove less significant and correlated features for better accuracy
        self.removeCorrelFeatures(X)
        self.removeInsigFeatures(X, Y)

        #normalize data for better convergence and to prevent overflow
        normalizedX = MinMaxScaler().fit_transform(X.values)
        X = pd.DataFrame(normalizedX)

        # insert a 1 in every row for intercept
        X.insert(loc=len(X.columns), column='intercept', value=1)

        # split data into test set and train sets for classification
        xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.2, random_state=42)

        # call SGD for training the data
        W = self.stocasGradDescent(xTrain.to_numpy(), yTrain.to_numpy())

        # begin setting predicted by testing model with dot loop
        yPredicted = np.array([])
        # loop through predicted
        for i in range(xTrain.shape[0]):
            yp = np.sign(np.dot(xTrain.to_numpy()[i], W))
            yPredicted = np.append(yPredicted, yp)
        # determine any predicitons
        yTestPredicted = np.array([])
        #loop through xTest and dot
        for i in range(xTest.shape[0]):
            yp = np.sign(np.dot(xTest.to_numpy()[i], W))
            yTestPredicted = np.append(yTestPredicted, yp)

        # print final accuracy and precision
        print("Accuracy of prediction on data:",'{0:.2f}%'.format(accuracy_score(yTest, yTestPredicted) * 100))
        print("Precision of predictor on data:",'{0:.2f}%'.format(recall_score(yTest, yTestPredicted) * 100))

    def calcCostGradient(self, W, xBatch, yBatch):
        #if only one example is passed (eg. in case of stocasGradDescent)
        if type(yBatch) == np.float64:
            yBatch = np.array([yBatch])
            xBatch = np.array([xBatch])
            #gives multidimensional array

        distance = 1 - (yBatch * np.dot(xBatch, W))
        dw = np.zeros(len(W))
        #Calculate weight (cost) values throughout data points
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.iterations * yBatch[ind] * xBatch[ind])
            dw += di
        #Average weight
        dw = dw/len(yBatch)
        return dw

    # function to apply stochastic gradient descent to optimize objective function.
    def stocasGradDescent(self, features, outputs):
        maxIterations = 626
        weights = np.zeros(features.shape[1])
        nth = 0
        previousCost = float("inf")
        costThreshold = 0.01 # threshold for cost
        #Start Stochastic Gradient Descent
        for iteration in range(1, maxIterations):
            #shuffle feature and ouputs to prevent repeats
            X, Y = shuffle(features, outputs)
            for ind, x in enumerate(X):
                # calculate the cost gradient and set weights using learning rate and gradient
                ascent = self.calcCostGradient(weights, x, Y[ind])
                weights = weights - (self.learningRate * ascent)

            # convergence check on 5^nth iteration
            if iteration == 5 ** nth or iteration == maxIterations - 1:
                cost = self.computeCost(weights, features, outputs)
                print("Iteration: {}, Cost: {}".format(iteration, cost))
                # iteration stoppage threshold check
                if abs(previousCost - cost) < costThreshold * previousCost:
                    return weights
                previousCost = cost
                nth += 1
        return weights

    # function to compute hinge loss for the classifier
    def computeCost(self, W, X, Y):
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        # if distance is negative, default it to 0
        distances[distances < 0] = 0
        # hingeloss is equal to the number of iterations times the sum of distances over the number of datapoints
        hingeLoss = self.iterations * (np.sum(distances) / N)

        # find final cost given hinge loss
        cost = 1 / 2 * np.dot(W, W) + hingeLoss
        return cost

    # will run through the X values and determine if the given features are correlated 
    def removeCorrelFeatures(self, X):
        # set threshold to 90%
        corr_threshold = 0.9
        corr = X.corr()
        # set dropped columns to correlated shape features
        drop_columns = np.full(corr.shape[0], False, dtype=bool)
        # iterate throgh the features, if correlation is above threshold, drop columns
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= corr_threshold:
                    drop_columns[j] = True
        columns_dropped = X.columns[drop_columns]
        # drop columns officially and return them
        X.drop(columns_dropped, axis=1, inplace=True)
        return columns_dropped

    # removes insignificant features found in both regressed x and y columns. 
    # if found, the respective columns are dropped
    def removeInsigFeatures(self, X, Y):
        sl = 0.05
        regression_ols = None
        # create array to hold dropped, insignificant features
        columns_dropped = np.array([])
        for itr in range(0, len(X.columns)):
            # call ordinary least squares to fit and remove insignificant data
            regression_ols = sm.OLS(Y, X).fit()
            #set the max as a result of the fit
            max_col = regression_ols.pvalues.idxmax()
            max_val = regression_ols.pvalues.max()
            # if max val result is over threshold, remove insignif.
            if max_val > sl:
                X.drop(max_col, axis='columns', inplace=True)
                columns_dropped = np.append(columns_dropped, [max_col])
            else:
                break
        regression_ols.summary()
        return columns_dropped


