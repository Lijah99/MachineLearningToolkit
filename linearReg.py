import numpy as np
import utils

class LinearRegression:
   def fit(self,x,y):
       #reshape the input axes
       x = np.array(x).reshape(-1, 1)
       y = np.array(y).reshape(-1, 1)
       #grab dimensions
       x_shape = x.shape
       
       variableNum = x_shape[1]
       weights = np.random.normal(0, 1, (variableNum, 1) )
       intercept = np.random.rand(1)

        #iterate 500 times for linear regression
       for i in range(500):
           costWeight = np.sum(np.multiply(((np.matmul(x, weights) + intercept) - y), x)) * 2 / x_shape[0]
           costDC = np.sum(((np.matmul(x, weights) + intercept) - y)) * 2 / x_shape[0]
           weights -= 0.1 * costWeight
           intercept -= 0.1 * costDC
       return weights, intercept
