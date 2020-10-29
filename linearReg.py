import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
class LinearRegression:
   def fit(self,X,Y):
       X=np.array(X).reshape(-1,1)
       Y=np.array(Y).reshape(-1,1)
       
       x_shape = X.shape
       
       num_var = x_shape[1]
       weight_matrix = np.random.normal(0,1,(num_var,1))
       intercept = np.random.rand(1)
       for i in range(50):
           dcostdm = np.sum(np.multiply(((np.matmul(X,weight_matrix)+intercept)-Y),X))*2/x_shape[0]
           dcostdc = np.sum(((np.matmul(X,weight_matrix)+intercept)-Y))*2/x_shape[0]
           weight_matrix -= 0.1*dcostdm
           intercept -= 0.1*dcostdc
       return weight_matrix,intercept
#print(df.drop(['Gender'],axis=1))
reg = LinearRegression()
x = (df['Weight']-df['Weight'].mean())/df['Weight'].std()
y = (df["Height"]-df['Height'].mean())/df["Height"].std()
params = reg.fit(x,y)
plt.scatter(x[:180],y[:180])
pred = np.matmul(np.array(x[:180]).reshape(-1,1),params[0])+params[1]
plt.plot(x[:180],pred)
print(params)