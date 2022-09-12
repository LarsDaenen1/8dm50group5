import numpy as np
import matplotlib.pyplot as plt

def pred(x, w):
    """
    Predicts the output y based on the input x and fitted weights w 
    :param X: Input data matrix
    :param w: fitted weights
    :return: Predictions
    """
    x = np.insert(x, 0, 1, axis=1)
    # Calculate the predictions using weights
    y_pred = x.dot(w)
    
    return y_pred

def calcMSE(y, y_pred):
    """
    Calculates the mean-squared error
    :param y: true output
    :param y_hat: predicted output:
    :return: mean-square error
    """
    # Calculate the mean-squared error
    MSE = np.sum((y-y_pred)**2)/len(y)
    
    return MSE 

def kNNclassifier(X_train, X_test, y_train, k):
    """
    Implementation of k-Nearest neighbours classifiers
    :param X_train: Input training data
    :param y_train: Output training data
    :param X_test: Input test data
    :k: # of neighbours used for classification
    :return: predicted class
    """
    classes = []
    # Loop over all samples in the test set
    for i in range(len(X_test)):
        distances = []
        # Loop over all samples in the training set
        for j in range(len(X_train)):
            # Calculate the Euclidian distance between test set sample and training set sample
            distance = np.linalg.norm(X_train[j]-X_test[i])
            distances.append([distance, y_train[j]])
        # Sort based on distances
        distances_sorted = sorted(distances, key=lambda x: x[0])
        # Calculate the median predicted class based on first k neighbours
        classes.append(round(np.median(list(distances_sorted[i][1] for i in range(k)))))
        
    return classes

def kNNRegression(X_train, X_test, y_train, k):
    """
    Implementation of k-Nearest neighbours regression
    :param X_train: Input training data
    :param y_train: Output training data
    :param X_test: Input test data
    :param k: # of neighbours used for classification
    :return: predicted value
    """
    y_pred = []
    # Loop over all samples in the test set
    for i in range(len(X_test)):
        distances = []
        # Loop over all samples in the training set
        for j in range(len(X_train)):
            # Calculate the Euclidian distance between test set sample and training set sample
            distance = np.linalg.norm(X_train[j]-X_test[i])
            distances.append([distance, y_train[j]])
        # Sort based on distances
        distances_sorted = np.array(sorted(distances, key=lambda x: x[0]))
        # Calculate the mean predicted value based on first k neighbours
        y_pred.append(np.mean(distances_sorted[:k, 1]))
        
    return y_pred


def class_conditional_prob(X,Y):
    
    # Calculate probability of y=1
    Py = np.sum(Y[:,0])/Y.shape[0]

    # Loop over all features of X
    for i in range(X.shape[1]):
        X_feature = X[:, i]
        # Filter the cases where y=Y
        XY = X_feature[ Y[:,0] == 1 ]

        # Calculate the mean and sigma^2
        mu = np.mean(XY)
        var = np.std(XY)**2

        P = []
        
        # Calculate P(x=X|y=Y) 
        for x in np.linspace(0,1,100):
            Pxy = np.sqrt(1/(2*np.pi*var))*np.exp(-1/(2*var)*(x-mu)**2)
            Pcond = Pxy / Py
            P.append(Pcond)

        # Visualize the results
        plt.plot(list(np.linspace(0,1,100)), P, 'b-')
        plt.title('Class conditional probability feature {:.0f}'.format(i+1))
        plt.show()