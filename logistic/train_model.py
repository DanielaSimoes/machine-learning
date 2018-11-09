import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
import numpy as np


# # ______________________________ Logistic Regression _____________________________________ #
def logistic(wine_set):
    X = wine_set[:, 0:11]
    y = wine_set[:, 11]

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

    # 30% of the dataset is used for testing and 70% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)

    logistic = LogisticRegression()
    penalty = ['l1', 'l2']
    C = np.logspace(0, 4, 10)
    hyperparameters = dict(C=C, penalty=penalty)

    clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

    best_model = clf.fit(X_train, y_train)

    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])

    best_model.save('wine-model.h5')


if __name__ == '__main__':
    dataset = numpy.loadtxt("../winequality-white.csv", delimiter=";", skiprows=1)
    logistic(dataset)