import time
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense


def optimize(x_train, y_train):

    dense_layers = [0, 1, 2]
    layer_sizes = [32, 64, 128]
    classifications = 3

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

                model.fit(x_train, y_train, epochs=1000, batch_size=50, verbose=2, callbacks=[tensorboard])
