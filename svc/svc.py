from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tunning_hyperparameters import svc_tunning
from train_model import svc_train
from test_model import test
from evaluate_model import evaluate
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

    # Now we're going to Tunning Hyperparameters only with Train Data
    #best_parameters = svc_tunning(x_train, y_train)
    best_parameters = {'C': 10, 'gamma': 0.001 , 'kernel': 'rbf'}
    #best_parameters = {'C': 10, 'gamma': 0.9 , 'kernel': 'rbf'}

    # Pass the best parameters to train, and the Train Data
    trained_model = svc_train(x_train, y_train, c=best_parameters["C"], gamma=best_parameters["gamma"], kernel=best_parameters["kernel"])

    # Evaluate the model
    print("Evaluate model\n")
    evaluate(trained_model, x_train, y_train)

    # Test the model
    print("Evaluate model\n")
    test(trained_model, x_test, y_test)

