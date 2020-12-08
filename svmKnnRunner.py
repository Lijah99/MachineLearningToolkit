import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

import svm
import knn


#Main Script Run
#This script will show as an example of the use of a KNN and SVM learners
#Created by Elijah Flinders

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)

#svm setup and training
print("Runnin the SVM on it's own dataset...")
svm = svm.SVM()
svm.calcSVM()
print("Finished running the SVM!\n")

#KNN setup and prediction
print()
print("Creating and testing KNN....")
knnTester = knn.KNN()

# load the Iris data set and convert to specific type
dataset = knnTester.loadCsvList('iris.csv')
for i in range(len(dataset[0]) - 1):
    knnTester.colToFloat(dataset, i)

# convert columns to ints
knnTester.colToInt(dataset, len(dataset[0]) - 1)

# define number of model neighbors and set record
neighbors = 5
testRow = [4.5, 2.3, 1.3, 0.3]

# predict the label
label = knnTester.predict(dataset, testRow, neighbors)
print('Data to form a prediction=%s, Predicted membership: %s\n' % (testRow, label))
