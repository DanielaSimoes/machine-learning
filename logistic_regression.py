import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# # ______________________________ Logistic Regression _____________________________________ #
def logistic(wine_set):

    X = wine_set[['quality']]

    y = wine_set[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
	
	
    # 30% of the dataset is used for testing and 70% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LogisticRegression()
    model = model.fit(y_train, X_train)

    predicted_data = model.predict(y_test)
    print("Mean squared error:", mean_squared_error(X_test, predicted_data))
    print("Coefficient of determination:", r2_score(X_test, predicted_data)*100,"%")

if __name__ == '__main__':
    white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';')
    logistic(white)