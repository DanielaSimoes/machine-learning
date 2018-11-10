from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.cross_validation import train_test_split
import numpy
from keras import optimizers



def nn_train(x_train, y_train):

    classifications = 3

    model = Sequential()
    model.add(Dense(12, input_dim=11, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(classifications, activation='softmax'))

    adam = optimizers.adam(lr=0.01, clipnorm=1.)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=500, batch_size=50, verbose=2, validation_data=(x_test, y_test))

    model.save('wine-model.h5')
