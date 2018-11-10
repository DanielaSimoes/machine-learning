

def evaluate(model, x_test, y_test):

    scores = model.evaluate(x_test, y_test)

    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
