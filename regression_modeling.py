import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn
import statsmodels.formula.api as smf


# # ______________________________ Linear Regression_____________________________________ #
def linear(wine_set):

    X = wine_set[['quality']]

    y = wine_set[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LinearRegression()
    model = model.fit(y_train, X_train)

    predicted_data = model.predict(y_test)
    print(mean_squared_error(X_test, predicted_data))


if __name__ == '__main__':
    white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';')
    linear(white)