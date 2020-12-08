import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
# learner imports
import svm
import knn


#Main Script Run
#This script will show as an example of the use of a KNN and SVM learners
#Created by Elijah Flinders

#svm setup and training
print("*********************************************************************")
print("Creating and testing SVM on it's own dataset. Support Vector Machine")
print("*********************************************************************")
svm = svm.SVM()
svm.fit()
print("Finished running the SVM!\n")

#KNN setup and prediction
print("************************************************")
print("Creating and testing KNN. K-th Nearest Neighbor")
print("************************************************")
knnTester = knn.KNN()

# load the Iris data set and convert to specific type
dataset = knnTester.loadCsvListKnn('iris.csv')
for i in range(len(dataset[0]) - 1):
    knnTester.colToFloat(dataset, i)

# convert columns to ints
knnTester.colToInt(dataset, len(dataset[0]) - 1)

# define number of model neighbors and set record
neighbors = 5
testSetosa = [4.5, 2.3, 1.3, 0.3]
testVersicolor = [7.0, 3.2, 4.7, 1.4]
testVirginica = [6.3, 3.3, 6.0, 2.5]
# predict the label
label = knnTester.predict(dataset, testSetosa, neighbors)
print('\nPrediction being fed to KNN=%s\nPredicted label/membership: %s. It was Setosa(2)' % (testSetosa, label))

label = knnTester.predict(dataset, testVersicolor, neighbors)
print('\nPrediction being fed to KNN=%s\nPredicted label/membership: %s. It was Versicolor(0)' % (testVersicolor, label))

label = knnTester.predict(dataset, testVirginica, neighbors)
print('\nPrediction being fed to KNN=%s\nPredicted label/membership: %s. It was Virginica(1)' % (testVirginica, label))

print("If values are off, try running the program again to give the trainer another shot!")
