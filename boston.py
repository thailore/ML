import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer

boston = load_boston()
'''
:Attribute Information (in order):
- CRIM     per capita crime rate by town
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    proportion of non-retail business acres per town
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      nitric oxides concentration (parts per 10 million)
- RM       average number of rooms per dwelling
- AGE      proportion of owner-occupied units built prior to 1940
- DIS      weighted distances to five Boston employment centres
- RAD      index of accessibility to radial highways
- TAX      full-value property-tax rate per $10,000
- PTRATIO  pupil-teacher ratio by town
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT    % lower status of the population
- MEDV     Median value of owner-occupied homes in $1000's

:Missing Attribute Values: None
'''


def performance_metric(y_true, y_predict):
    return r2_score(y_true, y_predict)


def fit_model(X, y):
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # Create a decision tree regression object
    regression = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(estimator=regression, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


def main():
    """
    X is features, y is target, df is DataFrame of all data
    """

    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    y = boston.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    print_price_stats(y)
    coefficient_matrix = np.corrcoef(df.values.T)

    display_price_coefficients(coefficient_matrix, df)

    model = fit_model(X_train, y_train)

    model_training_and_testing_stats(X_test, X_train, model, y_test, y_train)

    test_house = [[0.00632, 18.0, 2.31, 0.0, 0.538, 4.575, 65.2, 4.0900, 1.0, 296.0, 15.3, 296.90, 4.98]]
    print("Prediction: ${}".format(model.predict(test_house) * 1000))

    print("Parameter 'max_depth' is {} for the optimal model.".format(model.get_params()['max_depth']))


def model_training_and_testing_stats(X_test, X_train, model, y_test, y_train):
    true_value = y_train
    predicted_value = model.predict(X_train)
    accuracy = r2_score(true_value, predicted_value)
    true_value_test = y_test
    predicted_value_test = model.predict(X_test)
    accuracy_test = r2_score(true_value_test, predicted_value_test)
    training_error_train = mean_absolute_error(true_value, predicted_value)
    training_error_test = mean_absolute_error(true_value_test, predicted_value_test)
    print("Model accounts for {} of training data with deviation of {}".format(accuracy, training_error_train))
    print("Model accounts for {} of test data with deviation of {}".format(accuracy_test, training_error_test))


def display_price_coefficients(coefficient_matrix, df):
    price_coefficients = {}
    count = 0
    for feature in coefficient_matrix:
        if count > 12:
            break
        price_coefficients[boston.feature_names[count]] = feature[-1]
        count += 1
    sns.set(font_scale=1.5)
    sns.heatmap(coefficient_matrix,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 15},
                yticklabels=df.keys(),
                xticklabels=df.keys())
    # Uncomment to display coefficient matrix
    # plt.show()

    print("Price Coefficients:")
    for key, value in price_coefficients.items():
        print("{} :: {}".format(key, value))

    return price_coefficients


def print_price_stats(y):
    minimum_price = np.amin(y) * 1000
    maximum_price = np.amax(y) * 1000
    mean_price = np.mean(y) * 1000
    median_price = np.median(y) * 1000
    std_price = np.std(y) * 1000
    print("Statistics for Boston housing dataset:\n")
    print("Minimum price: ${}".format(minimum_price))
    print("Maximum price: ${}".format(maximum_price))
    print("Mean price: ${}".format(mean_price))
    print("Median price ${}".format(median_price))
    print("Standard deviation of prices: ${}\n".format(std_price))


if __name__ == '__main__':
    main()
