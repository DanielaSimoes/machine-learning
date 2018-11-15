

def evaluate(model, x_train, y_train):
    score = model.evaluate(x_train, y_train, batch_size=32)
    print(score)
