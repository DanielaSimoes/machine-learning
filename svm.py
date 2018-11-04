import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder




# # ______________________________ SVM  _____________________________________ #
def svm(df):

    X = df.drop(['quality'], axis=1)
    y = df['quality']

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # 30% of the dataset is used for testing and 70% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = SVC(kernel = 'rbf', random_state = 0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)

    print("Average accuracy: ", accuracies.mean())
    print("Standard deviation: ", accuracies.std())
    print("Wait a moment...")

    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],
                   'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

    grid_search = GridSearchCV(estimator = clf, param_grid = parameters, scoring = 'accuracy', cv = 10,)
    grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    print("Best accuracy: ", best_accuracy)
    print("Best parameters: ", best_parameters)

    print("Done")
    """
        Then we need to train with the best parameters and test again
    """

if __name__ == '__main__':
    white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';')
    svm(white)