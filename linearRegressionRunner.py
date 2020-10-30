import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ML import Perceptron
from ML import plot_decision_regions
from linearReg import LinearRegression

#Main Script Run
#Created by Elijah Flinders

#grab csv
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')

#Linear regression runner
reg = LinearRegression()
#Perceptron
pn = Perceptron(0.1,10)

#setting the indpended variable to price and the dependent to carat
x = (df['price']-df['price'].mean())/df['price'].std()
y = (df['carat']-df['carat'].mean())/df['carat'].std()
#fit the linear regression prediction line to the graph
params = reg.fit(x,y)
#add the data to the plot
plt.scatter(x[:53940],y[:53940], s = 1)
pred = np.matmul(np.array(x[:53940]).reshape(-1,1),params[0])+params[1]
plt.plot(x[:53940],pred, color = 'red')

#set plot title and axes
plt.title('Linear Regression: Diamond Data Set')
plt.xlabel('Price')
plt.ylabel('Carat')
plt.show()

#pn.fit(x,y)