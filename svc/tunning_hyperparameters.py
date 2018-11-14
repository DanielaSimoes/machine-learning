from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np


# # ___________________ SVC HyperParameters Tunning __________________ #
def svc_tunning(X_train, y_train):

    clf = SVC()

    parameters = [{'C': np.logspace(-3, 2, 6), 'kernel': ['rbf'],
                   'gamma': np.logspace(-3, 2, 6)}]

    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=-1,scoring='accuracy', cv=5,)

    grid_search.fit(X_train, y_train)

    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    print("Best accuracy: ", best_accuracy)
    print("Best parameters: ", best_parameters)

    return best_parameters