# MachineLearningToolkit: Elijah Flinders, 2020
A personally created library which allows use for machine learning purposes and creation of learners.

## Classifiers
- Perceptron
- Decision Stub 
- KNN - Kth Nearest Neighbor
- SVM - Support Vector Machine

## Runners
- Perceptron Runner: Will showcase a perceptron classifier
- Linear Regression Runner: Will showcase linear regression and display a Decision Stub Learner
- SvmKnnRunner: Will showcase the Kth Nearest Neighbor classifier and support vector machine
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## The Creator
This repository of machine learning algorithms was made by Elijah Flinders, a student at the South Dakota School of Mines. The creation of this portfolio of sorts was both a result of the machine learning class I decided to take, but also my general curiosity of machine learning algorithms in general.

## Portfolio Description
This is a portfolio containing cumulative ideas gathered in a semester of learning for a machine learning class. It is to display the progress and understanding of basic machine learning concepts that have been gathered throughout the semester. This library contains the following classifiers and runners to demonstrate them:

### Decision Stump -               Class: decisionStump.py Runner: linearRegStumpRunner.py
### Perceptron -                      Class: perceptron.py Runner: perceptronPractice.py
### K-th Nearest Neighbor -   Class: knn.py Runner: svmKnnRunner.py
### Support Vector Machine - Class: svm.py Runner: svmKnnRunner.py
### Linear Regression -          Class: LinearReg.py Runner: linearRegStumpRunner.py

Provided below is external library information, compilation/running instructions, and in-depth descriptions of classifiers and their respective functions.

## Uses of the Library
As stated, this entire library is a fairly comprehensive machine learning library with numerous learners and example runners that show them in action. 

The classifiers thus far are a decision stump class, a low level learning algorithm, a perceptron, a fairly high level learner, and both a k-th nearest neighbor and support vector machine algorithm. These are all relatively low VC. But as a beginner’s tool for classification and learners, it does the trick. For the most part, these classifiers can be modified or are set to use different datasets but a few were provided for example and educational purposes. 

Along with the classifiers is a utility file, utils.py, which holds functions useful to displaying the results of the classifiers themselves.

Finally, there are runner files, many of which will be used to run this exam. There is currently one for running and testing the functionality of the perceptron, called perceptronRunner.py. Another, which will be used for this project is linearRegressionRunner.py, which will showcase both linear regression, and the decision stub. Finally, there is the 

## Libraries Used or Created
Python Version- 3.7. Complied by the following command: python3 [scriptName].py

Numpy - Numerical Python library which allows for creation of multidimensional arrays the application of mathematical and logical operations on them.

Pandas - A Data Analysis python library, which is used to import and manipulate the data given in the iris flower dataset. It is the most pivotal library for getting data into a form that the algorithms and learners can use to predict and fit.

Matplotlib - A general mathematical plotting library used to create, display, and alter information to be displayed on graphs. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits

Utils - A personally created utility library which contains 2 functions so far to aid in the mathematical applications of the learners and to display the classification of the data by any type of fitted learner
plot_decision_regions(X, y, classifier, title="Graph", resolution = 0.02) - This function will take the X and Y matrices of data and required values, a classifier algorithm class, and will plot the results
mode(y) - This function will take an array and return the mode value in the array

Sklearn - A broad machine learning library for python which includes modules ranging from full blown algorithms to small utilities to help customized learners. In this portfolio, it is used to help split training and test data in a quick and efficient manner for the knn and svm algorithms.

Statsmodels - A statistics and modeling python library which also includes the appropriate functions for conducting statistical tests and statistical data exploration. In my case, I use the regression OLS function which provides a thresholding linear regression model use on my data to assist the KNN algorithm’s accuracy and precision by weeding out insignificant figures from my X and Y columns.

Scipy - Another math and engineering library with multi purpose functionality. In my case, I use it to construct a utility library - utils.py - where I use it to create a mode function to return the mode of a provided set of data.

## Datasets Used
The iris machine learning csv dataset was used to showcase decision stumps, the perceptron, and k-th nearest neighbor. Here is the source for that dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data 
The breast cancer tumor dataset was used for the support vector machine algorithm. Here is the source for that dataset:
http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29

Most of the algorithms were designed around these two datasets, but most can be easily used with another dataset. If necessary, algorithms can be easily adapted in their initialization and fitting functions to accommodate for more complicated datasets.

## How to Run/Compile 
Run: python3 [insert runnerFile].py
The only caveat is that you must ensure that your runner file, if making your own, imports the packages for their associated learner as follows:
This can be run in any cmd or bash terminal with access to the necessary libraries and python3.
This current library was last developed using python 3.7.

## Classifiers and Respective Functions
## Decision Stub Algorithm (Low Level)
A decision stump is a machine learning model consisting of a one-level decision tree and it is pretty low level in terms of trainers. Meaning t is a decision tree with one internal node (the root) which is immediately connected to the terminal nodes (its leaves). A decision stump makes a prediction based on the value of just a single input feature.

In our case, it’s analyzing the feature of whether it is a versicolor or not, that is the Y parameter of the fit and predict functions. 

It reshapes the arrays and determines split based on error comparison to our actual split. Once found, it will predict a mesh split based on the split afterwards, which will then be used to plot the classifications. 

Because of it’s weak dimension property and small breadth, it fails to classify our data set very accurately.

fit(self, X, y) - The main fitting function for the decision stump classifier. First it determines it’s fitting value for classification attributes by grabbing the mode using the utils.py library. Using the mode, it sets the minimum error for decision making. It will then loop through the data, looking for the best resulting split of the data, i.e. the decision stump. Once found, it will set the best splitting variables to be used in the prediction function.

predict(self, X) - The prediction function for the decision stump classifier. After fitting, it will take the X values of the dataset and use the splitting variables set in the fit function to split the data using those values, completing the classification. Here is it splitting the values: 


## Perceptron
The perceptron classifier is more complex than something as low level as the decision stump. Over the course of multiple iterations it will build a binary classification of a given dataset. It is also known as a linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

In our example case, it looks through the iris.csv dataset and tries to build it’s binary classification between the setosa and versicolor group based on sepal and petal length.

fit(self, X, Y) - The main fitting function. Sets up a weight and misclassification array, then loops through a specified number of iterations and a particular learning rate, where it will generate a binary classification based on deviation of the data from the weight.

predict(self, X) - The prediction function. It will determine the net inputs of the data given the weights generated by the fit function. This set of inputs will thus be the binary classification.


## K-th Nearest Neighbor
The K-th Nearest Neighbor classifier is simple and supervised, mainly used in both regression and classification.. It's easy to implement and understand, but has a major drawback of becoming significantly slower as the size of that data in use grows.

Specifically, it is a lazy learner, because the function is only approximated locally and all normalizing data is necessary to achieve high accuracy in most cases. In this library’s KNN class, this is done using the Statsmodel library.

loadCsvListKnn(self, filename) - loads the filename, a csv, to be used by the classifier.

dataMaxMin(self, dataset) - finds the minimum and maximum values for each column in the data set and creates a list for the classifier.

normalData(self, dataset, minmax) - Takes the dataset cols and minmax value, then normalizes the data to be in the range of 0 - 1. 

euclidDistCalc(self, row1, row2) - Calculates the euclidean distance between two vectors.

neighborCalc(self, train, testRow, numNeighbors) - Takes the training data, a testing data vector, and the number of neighbors, and determines similar neighbors by finding the euclidean distance.

predict(self, train, testRow, numNeighbors) - The prediction function for the classifier. Takes the training data, the example testing row, and the specified number of neighbors. It then calls the neighbor calculation and determines the prediction by grabbing the maximum set of output values.

## Support Vector Machine
The Support Vector Machine is a versatile classifier. It can both perform linear classification and non-linear classification. It’s main goal is to find a hyperplane in Feature-dimensional space that distinctly classifies the data points. Here is a visual example of such:

In our case, we remove correlated and insignificant features, using a statsmodels regression model to increase the accuracy of the classification. The data is then split into train and test data and stochastic gradient descent is performed to optimize the objective function and determine the derivative from training data with test data and is done iteratively.

fit(self) - Main fit function for the support vector machine classifier. First it reads in the data provided, in this example it uses breastCancerData.csv, but can be easily modified. It then separates features and outputs, cleans up insignificant features, splits training a test data, then performs stochastic gradient descent.

stocasGradDescen(self, features, outputs) - Function that performs stochastic gradient descent on the determined features and outputs of the data. Requires the calculation of the summarized weight (cost) values throughout the data points and then divides them by the number of datapoints to obtain the average weight for the current batch of tested values. The iterations are set to a max of 626. Features are shuffled and weights are updated dby weights - the learning rate * the calculated cost gradient. There is a stoppage function that will check to see if the cost is decreasing at a sufficient rate, as to not waste the time of the leaner, and thus the user which may not want diminishing returns.

## Linear Regression
This class is able to generate a linear average given for a data set. It’s used when you want to predict the value of a variable based on the value of another variable. It’s very heavily used in machine learning and statistical modeling. An average is generated using the summation of all given data points divided into one. This average is then used to predict a linear average of the given dataset. Here is an example of linear regression in the iris dataset.

fit(self, x, y) - The main fit function for the linear regression class. It will take the X and Y of the dataset, generate weights and a random intercept. Then it will iterate 500 times, minimizing the cost weights of the intercept in the data to get an optimal linear regression fit on the data.
