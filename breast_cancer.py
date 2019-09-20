from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
print("Description: {}".format(cancer['DESCR']))
print("Keys: {}".format(cancer.keys()))
print("Data shape: {}".format(cancer.data.shape))

print("Sample Counts per target name: {}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

# try n neighbors from 1-10
neighbors_range = range(1, 11)

for n_neighbors in neighbors_range:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)

    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_range, training_accuracy, label="Training accuracy")
plt.plot(neighbors_range, test_accuracy, label="Test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("neighbors")
plt.legend()
plt.show()
