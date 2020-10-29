import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
#df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')

class LinearRegression:
   def fit(self,X,Y):
       X=np.array(X).reshape(-1,1)
       Y=np.array(Y).reshape(-1,1)
       
       x_shape = X.shape
       
       num_var = x_shape[1]
       weight_matrix = np.random.normal(0,1,(num_var,1))
       intercept = np.random.rand(1)
       for i in range(500):
           dcostdm = np.sum(np.multiply(((np.matmul(X,weight_matrix)+intercept)-Y),X))*2/x_shape[0]
           dcostdc = np.sum(((np.matmul(X,weight_matrix)+intercept)-Y))*2/x_shape[0]
           weight_matrix -= 0.1*dcostdm
           intercept -= 0.1*dcostdc
       return weight_matrix,intercept

# #Linear regression runner
# reg = LinearRegression()

# #setting the indpended variable to price and the dependent to carat
# x = (df['price']-df['price'].mean())/df['price'].std()
# y = (df['carat']-df['carat'].mean())/df['carat'].std()
# #fit the linear regression prediction line to the graph
# params = reg.fit(x,y)
# #add the data to the plot
# plt.scatter(x[:53940],y[:53940])
# pred = np.matmul(np.array(x[:53940]).reshape(-1,1),params[0])+params[1]
# plt.plot(x[:53940],pred, color = 'red')

# #set plot title and axes
# plt.title('Linear Regression: Diamond Data Set')
# plt.xlabel('Price')
# plt.ylabel('Carat')
# #print(params)
# plt.show()