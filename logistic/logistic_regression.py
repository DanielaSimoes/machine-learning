import numpy
from sklearn.cross_validation import train_test_split
from tunning_hyperparameters import logistic_tunning
from train_model import logistic_train
from evaluate_model import evaluate

if __name__ == '__main__':

    # Load Dataset
    dataset = numpy.loadtxt("../winequality-white.csv", delimiter=";", skiprows=1)

    # Parse Data into 3 Categories
    y = dataset[:, 11]
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

    # Divide Dataset: 20% Test and 80% Train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # Now we're going to Tunning Hyperparameters only with Train Data
    best_parameters = logistic_tunning(x_train, y_train)

    # Pass the best parameters to train, and the Train Data
    trained_model_regularized = logistic_train(x_train, y_train, penalty=best_parameters["model_regularized"]["Penalty"], C=best_parameters["model_regularized"]["C"], solver=best_parameters["model_regularized"]["Solver"], multi_class=best_parameters["model_regularized"]["MultiClass"], max_iter=1000)
    trained_model_non_regularized = logistic_train(x_train, y_train, penalty=best_parameters["model_non_regularized"]["Penalty"], C=best_parameters["model_non_regularized"]["C"], solver=best_parameters["model_non_regularized"]["Solver"], multi_class=best_parameters["model_non_regularized"]["MultiClass"], max_iter=1000)

    # Evaluate the model
    print("Regularized: \n")
    evaluate(trained_model_regularized, x_test, y_test)
    print("Non Regularized: \n")
    evaluate(trained_model_non_regularized, x_test, y_test)
