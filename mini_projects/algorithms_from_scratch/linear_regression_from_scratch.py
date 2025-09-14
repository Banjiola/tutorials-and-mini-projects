# load all libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Implementation of OLS
class OLS:
    def __init__(self,X_train,y_train) -> None:
        self.X_original = X_train
        self.X_train = X_train
        self.y_train = y_train
        self.estimated_parameters = None

    # i need to firstly add the np.ones to the dataset (X)
    def prepare(self):
        bias = np.ones(self.X_original.shape[0])
        self.X_train = np.column_stack((bias, self.X_original)) # this is important to keep model from changing after every run
        return self.X_train
        
    def train(self):
        """
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Feature matrix, where n_samples is the number of datapoints and n_features is the number of predictors/features.
        y_train : array-like, shape (n_samples,) or (n_samples, 1)
            Target output vector.
        
        Returns
        -------
        estimated_parameters : ndarray, shape (n_features,) or (n_features, 1)
            Estimated coefficients using Ordinary Least Squares.
        """
        # We use Moore-Penrose pseudoinverse to handle cases where X^TX might be singular
        # np.matmul(np.linalg.pinv(np.matmul(X.T,X)),np.matmul(X.T,y)) --> This is more explicit but verbose and might be numerically
        # unstable for edge cases.
        self.prepare() # this is already defined in the prepare method
        self.estimated_parameters = np.matmul(np.linalg.pinv(self.X_train), self.y_train)
        return self.estimated_parameters

#so to make predictions I need to matmul the estimated params by the features
    def predict (self, X_test):
        bias = np.ones(X_test.shape[0])
        X_test = np.column_stack((bias, X_test))
        self.predictions = np.matmul(X_test, self.estimated_parameters)
        return self.predictions


# Plot to show the convergence of a convex function
X = np.linspace(-10,10,100)
y = X**2
#dy/dx = 2x so when 2x = 0, critical point is when x == 0, same logic applies to all squared function
plt.figure(figsize=(6,4))
plt.plot(X,y)
plt.scatter(0,0, c= 'r', zorder = -5, label = "Critical Point")# which is the global minimum in this case
plt.grid()
plt.legend(loc = 'center')
plt.xlabel('X')
plt.ylabel('y')
plt.title("Convergence of A Convex Function e.g RSS")
plt.tight_layout()
plt.show()


# fetch dataset from uci 
auto_mpg = fetch_ucirepo(id=9) 
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 
  
# print some info 
print(auto_mpg.metadata['additional_info']['summary'])
  
# variable information 
print(auto_mpg.variables) 

# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 
# merge data together as we will need to remove null
df = pd.concat((X,y), axis= 1)

print(f"The shape of the data is {df.shape}")
print('='*30)
print(X.info())
# drop null entries
df = df.dropna()

# select X and y
X = df.drop(columns='mpg')
y = df['mpg']

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.4, random_state= 42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred1 = lin_reg.predict(X_test)


mse1 = mean_squared_error(y_test, y_pred1)
mse1
my_lin_reg = OLS(X_train= X_train, y_train=y_train)
my_lin_reg.train()
y_pred2 = my_lin_reg.predict(X_test=X_test)
mse2 = mean_squared_error(y_test, y_pred2)
print(f"The mean squared error of sklearn's linear reg is {mse1:.2f}")
print(f"The mean squared error of our implementation is {mse2:.2f}")

