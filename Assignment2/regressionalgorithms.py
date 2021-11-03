import numpy as np
import math

import MLCourse.utilities as utils
import script_regression as script
import matplotlib.pyplot as plt

import time

# - Baselines -
# -------------

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.mean = None

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0.5,
            'features': [1,2,3,4,5],
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)

        """Question 2 a) Pseudo inverse"""
        self.weights = np.linalg.pinv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

# ---------
# - TODO: -
# ---------

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({'regwgt': 0.5}, parameters)

    """ Question2 c) """

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        #Xless = Xtrain[:, self.params['features']]
        Xless = Xtrain
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        #Xless = Xtest[:, self.params['features']]
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest


""" Question2 d) """


class LassoRegression(Regressor):
    """
    Batch Gradient Decent with Lasso Regression
    """

    def __init__(self, parameters={}):

        self.params = utils.update_dictionary_items({
                'regwgt': 0.0,
                'regwgt': 0.01,
                'regwgt': 1.0
        }, parameters)

    # proximal methods
    def prox(self, weight, stepsize, regwgt):

        for i in range(weight.shape[0]):
            # print (weight[i])
            if weight[i] > regwgt * stepsize:
                self.weights[i] = weight[i] - regwgt * stepsize
                # print (self.weights[i])
            elif np.absolute(weight[i]) <= regwgt * stepsize:
                self.weights[i] = 0
                # print (self.weights[i])
            elif weight[i] < -regwgt * stepsize:
                self.weights[i] = weight[i] + regwgt * stepsize
                # print (self.weights[i])

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

        numsamples = Xtrain.shape[0]

        self.weights = np.zeros([385, ])  # intialize the weights of vectors of zeros
        # print (self.weights.shape)
        err = float('inf')  # set error to be infinite at the beginning
        tolerance = 10e-4
        # Dividing by numsamples before adding regularization
        # to make the regularization parameter not dependent on numsamples
        XX = np.dot(Xtrain.T, Xtrain) / numsamples
        Xy = np.dot(Xtrain.T, ytrain) / numsamples

        # Comput the Frobenius norm  of XX
        norm = np.linalg.norm(XX, ord='fro')

        stepsize = 1 / (2 * norm)  # fixed stepsize
        # c(w)
        c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain)

        """
        Batch gradient descent for l1 regularized linear regression, from Algorithm 4 in notes
        """
        while np.absolute(c_w - err) > tolerance:
            err = c_w
            # print (var, self.weights)
            # The proximal operator projects back into the space of sparse solutions given by l1
            var = np.add(np.subtract(self.weights, stepsize * np.dot(XX, self.weights)), stepsize * Xy)
            self.prox(var, stepsize, self.params['regwgt'])
            # print (self.weights)
            # calculate l1 norm
            norm = np.linalg.norm(self.weights, ord=1)
            # update C(w)
            c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain) + self.params['regwgt'] * norm
        # print (self.weights)

    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest


""" Question2 e) """

class SGD(Regressor):

    def __init__(self, parameters={}):
        self.params = {}  # subselected features
        #self.reset(parameters)
        self.numruns = 5
        self.yaxis = []#np.zeros(1000)
        self.xaxis = []
        self.time = []

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(385,) # intialize the weights of random vectors

        # t = 1
        epochs = 1000
        stepsize_init = 0.01

        # xaxis = np.zeros(1000)


        times = []
        start = time.time()

        # -------------

        """
        Stochastic gradient descent, from Algorithm 3 in notes
        """
        for i in range(epochs):

            # shuffle data points from 1, ..., numbsamples
            arr = np.arange(numsamples)
            np.random.shuffle(arr)
            for j in arr:
                gradient = np.dot(np.subtract(np.dot(Xtrain[arr[j]].T, self.weights), ytrain[arr[j]]), Xtrain[arr[j]])
                # print (gradient)
                stepsize = stepsize_init / (1 + i)  # decrease stepsize to converge
                self.weights = np.subtract(self.weights, stepsize * gradient)
                # print(self.weights)

            # store the Error for plotting
            self.yaxis.append(script.geterror(np.dot(Xtrain, self.weights), ytrain))
            times.append(time.time() - start)




        '''Plot Error vs Epoch for SGD'''
        #x = np.arange(2000)
        self.xaxis = np.arange(epochs)
        self.time = times
        #plt.plot(self.time, self.yaxis)
        #plt.show()
        plt.plot(self.xaxis, self.yaxis)
        plt.xlabel('Number of Epoches')
        plt.ylabel('Error')
        plt.title('Error VS Epoches for SGD')
        plt.show()

    def data(self):
        """ return the averaged error in y axis """
        return self.yaxis, self.time

    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest



""" Question2 f) """


class batchGD(Regressor):

    def __init__(self, parameters={}):
        self.params = {}
        #self.reset(parameters)
        self.numruns = 5
        self.yaxis = []
        self.xaxis = []
        self.time = []


    """
    Line search algorithm, from Algorithm 1 in notes
    """

    def lineSearch(slef, Xtrain, ytrain, weight_t, gradient, cost):

        #numsamples = Xtrain.shape[0]
        stepsize_max = 1.0
        t = 0.5  # stepsize reduces more quickly
        tolerance = 10e-7
        stepsize = stepsize_max
        weight = weight_t.copy()  # weight_t is self.weights
        obj = cost
        max_interation = 100
        i = 0

        # while number of backtracking iterations is less than maximum iterations
        while i < max_interation:
            weight = weight_t - stepsize * gradient
            cost = script.geterror(np.dot(Xtrain, weight), ytrain)
            # Ensure improvement is at least as much as tolerance
            if (cost < obj - tolerance):
                break
            # Else, the objective is worse and so we decrease stepsize
            else:
                # print ("else")
                stepsize = t * stepsize
            i = i + 1
        # If maximum number of iterations reached, which means we could not improve solution
        if i == max_interation:
            # print ("i")
            stepsize = 0
            return stepsize
        return stepsize

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(385,)  # intialize the weights of random vectors

        count = 0  # count the number of iterations before converge
        # print (w.shape)
        err = float('inf')
        tolerance = 10e-7
        stepsize = 0.01

        times = []
        start = time.time()

        c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain)

        """
        Batch gradient descent, from Algorithm 2 in notes
        """
        while np.absolute(c_w - err) > tolerance:
            err = c_w
            gradient = np.dot(Xtrain.T, np.subtract(np.dot(Xtrain, self.weights), ytrain)) / numsamples
            # The step-size is chosen by line-search
            stepsize = self.lineSearch(Xtrain, ytrain, self.weights, gradient, c_w)
            # print(self.weights)
            self.weights = self.weights - stepsize * gradient
            c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain)
            # print(self.weights)


            # store the Error for plotting
            self.yaxis.append(c_w)
            count = count + 1
            times.append(time.time() - start)


        self.xaxis = np.arange(count)
        self.time = times
        #plt.plot(self.time, self.yaxis)
        #plt.show()
        #plt.plot(self.xaxis, self.yaxis)
        #plt.show()




    def data(self):
        """ return the averaged error in y axis """
        return self.yaxis, self.time

    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest



"""Bonus Question (a)"""

class Momentum(Regressor):
    def __init__(self, parameters={}):
      self.params = {}

      """initialize moving average as vector of zeros"""
      self.m_t = np.zeros([385, ])
      self.beta = 0.9
      self.alpha = 0.01
      self.weights = np.zeros([385, ])

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros([385, ])
        t=0

        while (t<1000):					#till 1000 iterations
            t+=1
            for j in range(numsamples):
                g_t = np.dot(np.subtract(np.dot(Xtrain[j].T, self.weights), ytrain[j]), Xtrain[j])		#computes the gradient of the stochastic function

                """Update exponential moving average over gradient"""
                m_t = self.beta* self.m_t + (1-self.beta)*g_t	#updates the moving averages of the gradient
                m_cap = m_t/(1-(self.beta**t))		#calculates the bias-corrected estimates
                weight_prev = self.weights
                self.weights = weight_prev - (self.alpha*m_cap)	#updates the parameters


    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest



"""Bonus Question (b) ADAM"""

class ADAM(Regressor):

    def __init__(self, parameters={}):
        self.params = {}


        self.m_t  = np.zeros([385,])
        self.v_t = np.zeros([385,])

        self.alpha = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999  # initialize the values of the parameters
        self.epsilon = 1e-8


    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros([385, ])
        t=0
        conv = 0

        while (t<1000):					#till 1000 iterations
            t+=1
            for j in range(numsamples):
                g_t = np.dot(np.subtract(np.dot(Xtrain[j].T, self.weights), ytrain[j]), Xtrain[j])		#computes the gradient of the stochastic function
                m_t = self.beta_1* self.m_t + (1-self.beta_1)*g_t	#updates the moving averages of the gradient
                v_t = self.beta_2* self.v_t + (1-self.beta_2)*np.dot(g_t.T,g_t)	#updates the moving averages of the squared gradient
                m_cap = m_t/(1-(self.beta_1**t))		#calculates the bias-corrected estimates
                v_cap = v_t/(1-(self.beta_2**t))		#calculates the bias-corrected estimates
                weight_prev = self.weights
                self.weights = weight_prev - (self.alpha*m_cap)/(np.sqrt(v_cap)+self.epsilon)	#updates the parameters


    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest



"""Bonus Question (c)"""

class ADAM_WithoutBias(Regressor):

    def __init__(self, parameters={}):
        self.params = {}


        self.m_t  = np.zeros([385,]) #np.zeros([385,])
        self.v_t = np.zeros([385,]) #np.zeros([385,])

        self.alpha = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999  # initialize the values of the parameters
        self.epsilon = 1e-8


    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros([385, ])
        t=0
        conv = 0

        while (t<1000):					#till 1000 iterations
            t+=1
            for j in range(numsamples):
                g_t = np.dot(np.subtract(np.dot(Xtrain[j].T, self.weights), ytrain[j]), Xtrain[j])		#computes the gradient of the stochastic function

                """Update momentum's removing initialization bias"""

                m_t =  self.m_t + g_t	#removing initialization bias
                v_t =  self.v_t + np.dot(g_t.T,g_t)	#removing initialization bias
                m_cap = m_t		#no bias-corrected estimates as bias removed
                v_cap = v_t		#no bias-corrected estimates as bias removed
                weight_prev = self.weights
                self.weights = weight_prev - (self.alpha*m_cap)/(np.sqrt(v_cap)+self.epsilon)	#updates the parameters


    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest