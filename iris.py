import mglearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

print("X Test Shape: {}".format(X_test.shape))
print("y Test Shape: {}".format(y_test.shape))

print("X Train Shape: {}".format(X_train.shape))
print("y Train Shape: {}".format(y_train.shape))

''' plot chart '''
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
chart = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# Train model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Add new data to test
X_new = np.array([[5, 2.9, 1, 0.2], [7, 1.9, 2.3, 2.4]])
print("X new shape: {}".format(X_new.shape))

# Make prediction
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target names: {}".format(iris_dataset['target_names'][prediction]))

# Model accuracy
y_pred = knn.predict(X_test)
print("Model accuracy: {}".format(np.mean(y_pred == y_test)))

plt.show()
