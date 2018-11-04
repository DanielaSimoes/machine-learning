import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# # ______________________________ Linear Regression _____________________________________ #
def linear(wine_set):

    y = wine_set[['quality']]

    X = wine_set[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]

    # 20% of the dataset is used for testing and 80% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

    model = LinearRegression(normalize=True)
    model = model.fit(X_train, y_train)

    predicted_data = model.predict(X_test)
    params = model.get_params()

    print("Params: ", params)
    print("Mean squared error:", mean_squared_error(y_test, predicted_data))
    print("Coefficient of determination:", r2_score(y_test, predicted_data)*100,"%")


if __name__ == '__main__':
    white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';')
    linear(white)