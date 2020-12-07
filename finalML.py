import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#sklearn only used for logistic regression partial example
from sklearn.model_selection import train_test_split

import DecisionStumps
from DecisionStumps import plot_decision_regions
import Perceptron
import SVM
import knn
import LinearRegression
#import LogisticRegression
#read in the iris flower data set
df = pd.read_csv('iris.csv', header = None)

#create perceptron-compatible Y
perceptronY = df.iloc[0:100,4].values
perceptronY = np.where(perceptronY == 'Iris-setosa',-1,1)
#create perceptron-compatible X
perceptronX = df.iloc[0:100, [2,3]].values

#Create a decisionStump
print()
print("Creating and testing the decision stump (low level learner):")
test = DecisionStumps.DecisionStump()
#fit it
test.fit(perceptronX,perceptronY)
#plot decisionstump
plot_decision_regions(perceptronX, perceptronY, test)
print()

#create perceptron object to create hypothesis
print()
print("Creating and displaying the perceptron hypothesis (smarter than previous decision stump):")
pn = Perceptron.Perceptron(0.1,10)
#fit it
pn.fit(perceptronX,perceptronY)
#plot hypothesis
plot_decision_regions(perceptronX,perceptronY,pn)
print()

#create the linearregression object, and it's x and y values
print()
print("Creating and displaying the linear regression object:")
linearRegression = LinearRegression.LinearRegression()
#x values are petal length of iris dataset, while y values are petal width of iris dataset
x = (df[2]-df[2].mean())/df[2].std()
y = (df[3]-df[3].mean())/df[3].std()
#fit it and store
parameter = linearRegression.fit(x,y)
plt.scatter(x[:180],y[:180])
#create a prediction
prediction = np.matmul(np.array(x[:180]).reshape(-1,1),parameter[0])+parameter[1]
#plot with labels
plt.plot(x[:180],prediction)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()
print()

###
### NOT REQUIRED AND NOT PROPERLY GRAPHING
###
#create the Logistic regression object and display it as well
#print()
#print("Creating and displaying the logistic regression object (not required and not properly graphing):")
#Importing dataset, overriding the iris dataset
#df = pd.read_csv("diabetes.csv")
#grab new X and Y's
#X = df.iloc[:, :-1].values
#Y = df.iloc[:, -1:].values
#Splitting dataset into train and test set, setting test size to 1/3 of dataset
#xTrain, xTest, yTrain, yTest = train_test_split(X, Y, 1/3, 0)
#Model training with 100,000 iterations to improve accuracy since it is difficult to get accuracy > 50%
#model = LogisticRegression.LogisticRegression(0.01, 100000)
#model.fit(xTrain, yTrain)
#Prediction on test set
#yPrediction = model.predict(xTest)
#measure performance
#correctlyClassified = 0
#counter
#count = 0
#Begin loop to count number of correct classifications
#for count in range(np.size(yPrediction)):
    #Correct analysis if yTest is the same as our yPrediction generated
    #if yTest[count] == yPrediction[count]:
        #correctlyClassified = correctlyClassified + 1
    #count = count + 1
#print("Accuracy on test set by our model:  ", (correctlyClassified / count) * 100)
#print()

#SVM Portion
print()
print("Creating and Testing SVM:")
#Create a SVM object with default iterations and learning rate (10000 and 0.000001
svmTester = SVM.SVM()
#Program run code got too long so it is functionalized
svmTester.calcSVM()
print()

#KNN Time
print()
print("Creating and testing KNN:")
knnTester = knn.KNN()
# Make a prediction with KNN on Iris Dataset
dataset = knnTester.load_csv('iris.csv')
for i in range(len(dataset[0]) - 1):
    knnTester.strColumnToFloat(dataset, i)
# convert class column to integers
knnTester.strColumnToInt(dataset, len(dataset[0]) - 1)
# define model parameter
num_neighbors = 5
# define a new record
row = [4.5, 2.3, 1.3, 0.3]
# predict the label
label = knnTester.predict(dataset, row, num_neighbors)
print('Data to form a prediction=%s, Predicted membership: %s' % (row, label))
print()
