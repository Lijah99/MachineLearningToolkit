import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

class SVM:

    def __init__(self):
        # set default values of learning rate and iterations
        self.iterations = 10000
        self.learningRate = 0.000001

    def fit(self):
        #read data in using pandas
        data = pd.read_csv('breastCancerData.csv')

        #drop last column (extra column added by pd)
        #and unnecessary first column (id)
        data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

        #convert categorical labels to numbers
        #M-malignant tumor, B-benign tumor
        diagnosisMap = {'M': 1.0, 'B': -1.0}
        data['diagnosis'] = data['diagnosis'].map(diagnosisMap)

        #put features and outputs in different data axis'
        Y = data.loc[:, 'diagnosis']
        X = data.iloc[:, 1:]

        #remove less significant features for better accuracy
        self.removeCorrelFeatures(X)
        self.removeInsigFeatures(X, Y)

        #normalize data for better convergence and to prevent overflow
        normalizedX = MinMaxScaler().fit_transform(X.values)
        X = pd.DataFrame(normalizedX)

        #insert 1 in every row for intercept b
        X.insert(loc=len(X.columns), column='intercept', value=1)

        #split data into train and test set
        xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.2, random_state=42)

        #train the model
        W = self.sgd(xTrain.to_numpy(), yTrain.to_numpy())

        #testing the model
        yPredicted = np.array([])
        for i in range(xTrain.shape[0]):
            yp = np.sign(np.dot(xTrain.to_numpy()[i], W))
            yPredicted = np.append(yPredicted, yp)
        #make our predictions
        yTestPredicted = np.array([])
        for i in range(xTest.shape[0]):
            yp = np.sign(np.dot(xTest.to_numpy()[i], W))
            yTestPredicted = np.append(yTestPredicted, yp)

        print("Accuracy of prediction on data:",'{0:.2f}%'.format(accuracy_score(yTest, yTestPredicted) * 100))
        print("Precision of predictor on data:",'{0:.2f}%'.format(recall_score(yTest, yTestPredicted) * 100))

    def calcCostGradient(self, W, xBatch, yBatch):
        #if only one example is passed (eg. in case of SGD)
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

    def sgd(self, features, outputs):
        maxIterations = 626
        weights = np.zeros(features.shape[1])
        nth = 0
        previousCost = float("inf")
        #cost threshold in percentage
        costThreshold = 0.01
        #SGD officially begins
        for iteration in range(1, maxIterations):
            #shuffle to prevent repeating update cycles
            X, Y = shuffle(features, outputs)
            for ind, x in enumerate(X):
                ascent = self.calcCostGradient(weights, x, Y[ind])
                weights = weights - (self.learningRate * ascent)

            #convergence check on 2^nth iteration
            if iteration == 5 ** nth or iteration == maxIterations - 1:
                cost = self.computeCost(weights, features, outputs)
                print("Iteration: {}, Cost: {}".format(iteration, cost))
                #stoppage criterion
                if abs(previousCost - cost) < costThreshold * previousCost:
                    return weights
                previousCost = cost
                nth += 1
        return weights

    def computeCost(self, W, X, Y):
        # calculate hinge loss
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        # set distances below zero to zero
        distances[distances < 0] = 0
        # hingeloss is equal to the number of iterations times the sum of distances over the number of datapoints
        hingeLoss = self.iterations * (np.sum(distances) / N)

        # calculate cost
        cost = 1 / 2 * np.dot(W, W) + hingeLoss
        return cost

    # will run through the X values and determine if the features are correlated
    # if found, the corresponding columns are dropped
    def removeCorrelFeatures(self, X):
        corr_threshold = 0.9
        corr = X.corr()
        drop_columns = np.full(corr.shape[0], False, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= corr_threshold:
                    drop_columns[j] = True
        columns_dropped = X.columns[drop_columns]
        X.drop(columns_dropped, axis=1, inplace=True)
        return columns_dropped

    # removes insignificant features found in both regressed x and y columns. 
    # if found, the respective columns are dropped
    def removeInsigFeatures(self, X, Y):
        sl = 0.05
        regression_ols = None
        columns_dropped = np.array([])
        for itr in range(0, len(X.columns)):
            regression_ols = sm.OLS(Y, X).fit()
            max_col = regression_ols.pvalues.idxmax()
            max_val = regression_ols.pvalues.max()
            if max_val > sl:
                X.drop(max_col, axis='columns', inplace=True)
                columns_dropped = np.append(columns_dropped, [max_col])
            else:
                break
        regression_ols.summary()
        return columns_dropped


