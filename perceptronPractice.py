import pandas as pd, numpy as np, matplotlib.pyplot as plt
from perceptron import Perceptron
import utils

#Main Script Run
#This script will show as an example of the use of a Perceptron in Machine Learning
#Created by Elijah Flinders

#grab csv
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

#set axes info with csv
x = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

#set graph characteristics for sepal/petal
plt.scatter(x[:50, 0], x[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc = 'upper left')
plt.show()

#create and set perceptron, fit using x,y
pn = Perceptron(0.1, 10)
pn.fit(x, y)

#set graph characteristics for misclassifications
plt.plot(range(1, len(pn.misclass) + 1), pn.misclass, marker = 'o')
plt.xlabel('Iteration')
plt.ylabel('# of misclassifications')
plt.show()

#plot decision regions
utils.plot_decision_regions(x, y, pn)
