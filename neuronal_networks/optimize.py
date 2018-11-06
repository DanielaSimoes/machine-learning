import time
import numpy
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten
from sklearn.cross_validation import train_test_split


def optimize(dataset):
    dense_layers = [0, 1, 2]
    layer_sizes = [32, 64, 128]

    y = dataset[:, 11]
    classifications = 3

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

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
                NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
                print(NAME)

                model = Sequential()

                model.add(Dense(12, input_dim=11, activation="relu"))

                for _ in range(dense_layer):
                    model.add(Dense(layer_size, activation="relu"))

                model.add(Dense(8, activation='relu'))
                model.add(Dense(classifications, activation="softmax"))

                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                adam = optimizers.adam(lr=0.01, clipnorm=1.)
                model.compile(loss='sparse_categorical_crossentropy',
                              optimizer=adam,
                              metrics=['accuracy'],
                              )

                model.fit(x_train, y_train, epochs=1000, batch_size=50, verbose=2, validation_data=(x_test, y_test),
                          callbacks=[tensorboard])

if __name__ == '__main__':
    dataset = numpy.loadtxt("../winequality-white.csv", delimiter=";", skiprows=1)
    optimize(dataset)