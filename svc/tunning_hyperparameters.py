from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# # ___________________ SVC HyperParameters Tunning __________________ #
def svc_tunning(X_train, y_train):

    clf = SVC()

    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],
                   'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=-1,scoring='accuracy', cv=10,)

    grid_search.fit(X_train, y_train)

    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    print("Best accuracy: ", best_accuracy)
    print("Best parameters: ", best_parameters)

    return best_parameters