import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import normalize


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
    
    #X_norm = normalize(X,'l2')
    #print("X:",X[1,:])
    #print("X_norm:",X_norm[1,:])	
    # 30% of the dataset is used for testing and 70% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predicted_data = model.predict(X_test)

	
    print("Mean squared error:", mean_squared_error(y_test, predicted_data))
    print("Coefficient of determination:", r2_score(y_test, predicted_data))
    print("Accuracy:", accuracy_score(y_test, predicted_data))
    #print("Pred data:",predicted_data[0:9])
    #print("Real data:",y_test[0:9])	

if __name__ == '__main__':
    dataset = numpy.loadtxt("winequality-white.csv", delimiter=";", skiprows=1)
    logistic(dataset)