from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def evaluate(model, x_train, y_train):

    predicted_data = model.predict(x_train)

    print("Score - training set:", model.score(x_train, y_train))
    print("Mean squared error - training set:", mean_squared_error(y_train, predicted_data))
    print("Coefficient of determination - training set:", r2_score(y_train, predicted_data))
    print("Accuracy - training set:", accuracy_score(y_train, predicted_data))

