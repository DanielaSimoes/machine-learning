import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np


# # ______________________________ Linear Regression _____________________________________ #
def linear(wine_set):

    X = wine_set[:, 0:11]

    y = wine_set[:, 11]
    new_y = []

    for each in y:
        if 0 <= each <= 4:
            # 0, 1, 2, 3, 4
            new_y.append(0)
        elif 5 <= each <= 6:
            # 5, 6
            new_y.append(1)
        else:
            # 7, 8, 9, 10
            new_y.append(2)

    y = new_y

    # 20% of the dataset is used for testing and 80% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

    model = PolynomialFeatures(degree=4)
    x_ = model.fit_transform(X)
    x_test_ = model.fit_transform(X_test)

    lg = LinearRegression()
    lg.fit(x_, y)
    predicted_data = lg.predict(x_test_)
    predicted_data = np.round_(predicted_data)

    print("Mean squared error:", mean_squared_error(y_test, predicted_data))
    print("Coefficient of determination:", r2_score(y_test, predicted_data)*100,"%")


if __name__ == '__main__':
    dataset = np.loadtxt("winequality-white.csv", delimiter=";", skiprows=1)
    linear(dataset)