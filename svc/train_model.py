from sklearn.svm import SVC


def svc_train(x_test, y_test, c, gamma, kernel):

    clf = SVC(C=c, gamma=gamma, kernel=kernel)
    clf.fit(x_test, y_test)

    return clf

