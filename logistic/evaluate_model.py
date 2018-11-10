from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def evaluate(model, x_test, y_test):

    predicted_data = model.predict(x_test)

    print("Score:", model.score(x_test, y_test))
    print("Mean squared error:", mean_squared_error(y_test, predicted_data))
    print("Coefficient of determination:", r2_score(y_test, predicted_data))
    print("Accuracy:", accuracy_score(y_test, predicted_data))

