import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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
    
    # Find indices of shortest euclidian distances for all test samples to their neighbours by sorting the numpy array
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

    
    
    
def class_cond_prob(X, y):

    """
    Computes the probability density curves of the different 
    features of the input data conditional on the binary 
    class of output / target data y.
    Assumes Gaussian distribution
    """
    
    
    for i in range(X.shape[1]):

        # For each feature (i) we check what the values of that feature is for all samples that are classified as y = 1
        X_class1 = X[:, i][y==1]
        X_class0 = X[:, i][y==0]

        # We calculate the mean and std for the Gaussian distribution we can assume
        mu_0 = np.mean(X_class0)
        sigma_0 = np.std(X_class0)
        
        mu_1 = np.mean(X_class1)
        sigma_1 = np.std(X_class1)


        # We plot this distribution for each feature
        fig, ax = plt.subplots(1, 1)
        rv_0 = norm(mu_0, sigma_0)
        rv_1 = norm(mu_1, sigma_1)

        # Define x values based on mu and sigma

        x_0 = np.linspace(norm.ppf(0.01, loc=mu_0, scale=sigma_0),norm.ppf(0.99, loc=mu_0, scale=sigma_0), 100) 
        x_1 = np.linspace(norm.ppf(0.01, loc=mu_1, scale=sigma_1),norm.ppf(0.99, loc=mu_1, scale=sigma_1), 100) 
        
        # Plot the probability density functions conditional on y = 0 and y = 1
        ax.plot(x_0, rv_0.pdf(x_0), 'r-', lw=2, label='frozen pdf0')
        ax.plot(x_1, rv_1.pdf(x_1), 'b-', lw=2, label='frozen pdf1')
        legend = ['Conditional y = 0', 'Conditional y = 1']
        plt.legend(legend)
        plt.xlabel('feature_' + str(i) + ' values')
        plt.ylabel('Probability density')
        plt.title('Conditional probability density plots of feature_' + str(i), fontsize=10)












    
    
    