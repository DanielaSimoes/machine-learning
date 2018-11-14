from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from keras import losses
import numpy as np


def evaluate(model, x_train, y_train):
    y_train = np.array(y_train)
    score = model.evaluate(x_train, y_train, batch_size=32)
    print(score)
