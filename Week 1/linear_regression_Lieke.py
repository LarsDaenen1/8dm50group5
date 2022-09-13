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

def MSE(X_test, y_test, beta):
    """

    Parameters
    ----------
    X_test : numpy array
        An array with the features of the test dataset
    y_test : numpy array
        An array with the target variable.
    beta : float
        Estimated coefficient vector for the linear regression

    Returns
    -------
    MSE : float
        The mean squared error of the prediction.

    """
    
    #add column of ones for the intercept
    ones = np.ones((len(X_test), 1))
    X_test = np.concatenate((ones, X_test), axis=1)
    
    #predict y
    y_predict = X_test.dot(beta)
    
    #calculating the MSE
    MSE = (1/len(y_predict))*np.sum((y_predict-y_test)**2)

    return MSE
    
    