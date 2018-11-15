from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import cross_val_score


def test(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)
    #predicted_data = model.predict(x_test)

    #print("Score CV - test set:", cross_val_score(model, x_test, y_test, cv=10).mean())
    #print("Mean squared error - training set:", mean_squared_error(y_test, predicted_data))
    #print("Coefficient of determination - training set:", r2_score(y_test, predicted_data))
    #print("Accuracy - training set:", accuracy_score(y_test, predicted_data))


