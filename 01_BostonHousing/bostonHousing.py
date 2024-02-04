import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Loading the dataset from the CSV file
file_path = 'BostonHousing.csv'
data = pd.read_csv(file_path)

# Assuming X contains 'RM' feature and y contains the target variable 'MEDV' where,
#   rm: Average number of rooms per dwelling &
#   medv: Median value of owner-occupied homes
X = np.array(data['rm']).reshape(-1, 1)
y = np.array(data['medv'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and fitting the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model by mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean square error: {mse}')

# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.title('Actual vs. Predicted Values')
plt.show()