from keras.models import load_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


import numpy

if __name__ == '__main__':

    dataset = numpy.loadtxt("../winequality-white.csv", delimiter=";", skiprows=1)
    y = dataset[:, 11]
    classifications = 3

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

    X = dataset[:, 0:11]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    model = load_model('wine-model.h5')

    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    predicted_data = model.predict(x_test)

    print("Mean squared error:", mean_squared_error(y_test, predicted_data))
    print("Coefficient of determination:", r2_score(y_test, predicted_data))
    print("Accuracy:", accuracy_score(y_test, predicted_data))

