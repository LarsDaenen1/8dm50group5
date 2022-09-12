import numpy as np

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

def mse(y_pred, y_test):
    
    """
    Calculates the mean-squared error
    :param y: true output
    :param y_hat: predicted output
    :return: mean-square error
    """
    MSE = 1/(y_pred.shape[0])*np.sum((y_pred-y_test)**2)
    
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    