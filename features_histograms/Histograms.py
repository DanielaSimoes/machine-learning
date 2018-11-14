from sklearn.preprocessing import MaxAbsScaler
import numpy 
import matplotlib.pyplot as plt


def histograms(wine_set):

    X = wine_set[:, 0:11]
    y = wine_set[:, 11]
    features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
	
    #X = MaxAbsScaler().fit_transform(X)
    
    class0 = []
    class1 = []
    class2 = []
    i=0
    #feature=2
    kwargs = dict(histtype='bar', alpha=1, bins='auto', color='navy')
    plt.hist(y, **kwargs)
    plt.title('Quality')
    plt.ylabel('Counts')
    plt.xlabel('Value')
    plt.grid(axis='y', alpha=0.75)
    plt.show()   
	
	
	
    for feature in range(0,11):
        i=0
        class0 = []
        class1 = []
        class2 = []
        for each in y:
            if 0 <= each <= 4:
                class0.append(X[i,feature])
            elif 5 <= each <= 6:
                class1.append(X[i,feature])
            else:
                class2.append(X[i,feature])
            i += 1


        kwargs0 = dict(histtype='bar', alpha=0.5, bins='auto', color='red')
        kwargs1 = dict(histtype='bar', alpha=0.5, bins='auto', color='yellow')
        kwargs2 = dict(histtype='bar', alpha=0.5, bins='auto', color='navy')

        plt.hist(class0, **kwargs0)
        plt.hist(class1, **kwargs1)
        plt.hist(class2, **kwargs2)
        plt.title('Feature ' + str(feature+1) + ": " + features[feature])
        plt.ylabel('Counts')
        plt.xlabel('Value')
        plt.grid(axis='y', alpha=0.75)
        plt.legend(("Classification: 0","Classification: 1","Classification: 2"))
        plt.show()
    
	
	
	
	
if __name__ == '__main__':
    dataset = numpy.loadtxt("../winequality-white.csv", delimiter=";", skiprows=1)
    histograms(dataset)