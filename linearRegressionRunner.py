import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from ML import Perceptron
from ML import plot_decision_regions
from linearReg import LinearRegression

#Main Script Run
#Created by Elijah Flinders

#grab csv
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)

#Linear regression runner
regression = LinearRegression()
#Perceptron
perceptronClassifier = Perceptron(0.1,1000)

#setting the axes to sepal length and width
y = (data[0]-data[0].mean())/data[0].std()
x = (data[1]-data[1].mean())/data[1].std()

#fit the linear regression prediction line to the graph
params = regression.fit(x,y)

#add the data to the plot
plot.scatter(x[:180],y[:180], s = 15)
prediction = np.matmul(np.array(x[:180]).reshape(-1,1),params[0])+params[1]
plot.plot(x[:180],prediction, color = 'red')

#set plot title and axes
plot.title('Linear Regression: Iris Flower Set')
plot.xlabel('Sepal Length')
plot.ylabel('Sepal Width')
plot.show()

##creating and setting perceptron for classification of data based on x dimension
xPercep = data.iloc[0:180, [1,0]].values
yPercep = data.iloc[0:180, 4].values
yPercep = np.where(yPercep == 'Iris-setosa', -1, 1)

perceptronClassifier.fit(xPercep,yPercep)

#plot vc dim classification
plot_decision_regions(xPercep, yPercep, perceptronClassifier)

