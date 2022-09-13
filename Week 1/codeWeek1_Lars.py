import numpy as np
import matplotlib.pyplot as plt

def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta


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

def kNN(X_train, y_train, X_test, y_test, k, regression=False):
    
    """
    Implementation of k-Nearest neighbours classifiers
    :param X_train: Input training data
    :param y_train: Output training data
    :param X_test: Input test data
    :k: # of neighbours used for classification
    :return: predicted class
    or if the regression argument is set to 'True' the function will return the average value of the 
    k closest neighbours as prediction for the new data point
    """
    
    nsamples = X_train.shape[0]
    features = X_train.shape[1]

    # Broadcasting to find pairwise differences

    diff = X_train.reshape(nsamples,1,features)-X_test
    
    # Compute euclidian distance
    euc_dist = np.sqrt((diff**2).sum(2))
    euc_dist = euc_dist.T
    
    # Find indices of shortest euclidian distances to neighbours for all points by sorting the numpy array
    ind = np.argsort(euc_dist)

    # We select only the indices of the k closest points
    k_indices = ind[:,0:k]
    
    # We use those indices to find the corresponding target labels in the y_train
    k_targets = y_train[k_indices]
    
    if regression == False:
        median = np.round(np.median(k_targets, axis=1))
        return median

    else:
        mean = np.mean(k_targets, axis=1)
        return mean


def class_conditional_prob(X,Y):
    
    # Calculate probability of y=1
    Py = np.sum(Y[:,0] == 1)/Y.shape[0]

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
        for x in np.unique(XY):
            Pxy = np.sqrt(1/(2*np.pi*var))*np.exp(-1/(2*var)*(x-mu)**2)
            Pcond = Pxy / Py
            P.append(Pcond)

        # Visualize the results
        plt.plot(list(np.unique(XY)), P, 'b-')
        plt.title('Class conditional probability feature {:.0f}'.format(i+1))
        plt.show()