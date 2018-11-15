

def test(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)

