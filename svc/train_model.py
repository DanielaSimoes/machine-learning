from sklearn.svm import SVC


def svc_train(x_train, y_train, c, gamma, kernel):

    clf = SVC(C=c, gamma=gamma, kernel=kernel)
    clf.fit(x_train, y_train)

    return clf

