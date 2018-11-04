from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.cross_validation import train_test_split
import pandas as pd


if __name__ == '__main__':
    dataset = pd.read_csv('../winequality-white.csv', low_memory=False, sep=';')

    # number of classifications
    classifications = 10

    y = dataset[['quality']]

    X = dataset[
        ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]

    # 30% of the dataset is used for testing and 70% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

    y_train = keras.utils.to_categorical(y_train-1, classifications)
    y_test = keras.utils.to_categorical(y_test- 1, classifications)

    model = Sequential()
    model.add(Dense(64, input_dim=11, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(classifications, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=200,validation_data=(X_test, y_test), batch_size=50, verbose=2)

    model.save('wine-model.h5')
