import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Loading the dataset from the CSV file
df = pd.read_csv('PC2Dataset.csv')

# Assuming 'X' contains features and 'y' is the target variable
X = df.drop('c', axis=1)
y = df['c'].astype(int)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and fitting the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)

# Printing the accuracy value
print(f'Accuracy: {accuracy:.10f}')

# Visualizing the Decision Tree
plt.figure(figsize=(24, 14))
tree.plot_tree(model)
plt.savefig('decision_tree_plot.png')
plt.show()
