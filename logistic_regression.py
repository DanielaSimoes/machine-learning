import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# # ______________________________ Logistic Regression _____________________________________ #
def logistic(wine_set):

    y = wine_set[['quality']]

    X = wine_set[
        ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]

    # 30% of the dataset is used for testing and 70% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predicted_data = model.predict(X_test)

    print("Mean squared error:", mean_squared_error(y_test, predicted_data))
    print("Coefficient of determination:", r2_score(y_test, predicted_data)*100,"%")

if __name__ == '__main__':
    white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';')
    logistic(white)