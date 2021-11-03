import numpy as np

import MLCourse.utilities as utils


'''
Random predictor
'''
# Susy: ~50 error
class Classifier:
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        pass

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

'''
Linear Regression
'''
# Susy: ~27 error
class LinearRegressionClass(Classifier):
    def __init__(self, parameters = {}):
        self.params = {'regwgt': 0.01}
        self.weights = None

    def learn(self, X, y):
        # Ensure y is {-1,1}
        y = np.copy(y)
        y[y == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

'''
1. (a) Naive Bayes
'''

# Susy: ~25 error
class NaiveBayes(Classifier):
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = utils.update_dictionary_items({'usecolumnones': False}, parameters)


    def learn(self, Xtrain, ytrain):
        # obtain number of classes
        if ytrain.shape[1] == 1:
            self.numclasses = 2
        else:
            raise Exception('Can only handle binary classification')


        if self.params['usecolumnones'] == True:
            self.numfeatures = 9

            #Xtrain[:][self.numfeatures] = np.ones(Xtrain.shape)
        else:
            self.numfeatures = 8
            Xtrain = np.delete(Xtrain,-1,axis=1)


        #print('features', Xtrain.shape)


        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)


        ## Keep track of number of class lebel 0 and class lebel 1
        numC0 = 0
        numC1 = 0

        ## Separate the two inputs class based on the output
        for i in range(len(ytrain)):
            if ytrain[i] == 0.0:
                numC0 += 1
                self.means[0] = self.means[0] + Xtrain[i][:self.numfeatures]
            if ytrain[i] == 1.0:
                numC1 += 1
                self.means[1] = self.means[1] + Xtrain[i][:self.numfeatures]

        #calculate prior probability of y=1 or y=0
        self.p_0 = numC0 / len(ytrain)
        self.p_1 = numC1 / len(ytrain)

        ## Calculate the mean
        for i in range(self.numfeatures):
            self.means[0][i] = self.means[0][i] / numC0
            self.means[1][i] = self.means[1][i] / numC1

        ## Calculate the standard deviation
        for i in range(len(ytrain)):
            if ytrain[i] == 0.0:
                self.stds[0] += (Xtrain[i][:self.numfeatures] - self.means[0]) ** 2

            if ytrain[i] == 1.0:
                self.stds[1] += (Xtrain[i][:self.numfeatures] - self.means[1]) ** 2

        for i in range(self.numfeatures):
            self.stds[0][i] = (self.stds[0][i] / numC0) ** 0.5
            #print(self.stds[0][i])
            self.stds[1][i] = (self.stds[1][i] / numC1) ** 0.5

        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        predictions = []

        ytest = np.zeros(Xtest.shape[0], dtype=int)

        for i in range(numsamples):
            probC0 = 1
            probC1 = 1

            for j in range(self.numfeatures):
                probC0 = probC0 * utils.gaussian_pdf( Xtest[i][j], self.means[0][j], self.stds[0][j])
                probC1 = probC1 * utils.gaussian_pdf( Xtest[i][j], self.means[1][j], self.stds[1][j])

            probC0 = self.p_0 * probC0
            probC1 = self.p_1 * probC1
            # Determine which class give a better probability.
            if probC0 >  probC1:
                ytest[i] = 0.0
            elif probC1 >= probC0:
                ytest[i] = 1.0

        assert len(ytest) == Xtest.shape[0]
        return ytest



'''
1. (b) Logistic Reggression
'''

# Susy: ~23 error
class LogisticReg(Classifier):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({'stepsize': 0.01, 'epochs': 100}, parameters)
        self.weights = None



    def learn(self, X, y):

        self.weights = np.zeros(X.shape[1],)

        epochs = 100
        #stepsize = 0.01
        numsamples = X.shape[0]
        for i in range (epochs):
            # shuffle data points from 1, ..., numbsamples
            arr = np.arange(numsamples)
            np.random.shuffle(arr)
            for j in arr:
                #gradient = np.dot(np.subtract(np.dot(X[j].T,self.weights), y[j]), X[j])
                gradient = np.dot(np.subtract(utils.sigmoid(np.dot(X[arr[j]].T, self.weights)), y[arr[j]]), X[arr[j]])
                # print (gradient)
                self.stepsize = 0.01/(1+i)
                self.weights = self.weights-self.stepsize*gradient


    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0], dtype=int)

        # print (Xtest.shape, self.weights.shape)
        p_1 = utils.sigmoid(np.dot(Xtest, self.weights))
        for i in range (Xtest.shape[0]):
            if p_1[i] >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0


        assert len(ytest) == Xtest.shape[0]
        return ytest
'''
1. (c) Neural Network
'''
# Susy: ~23 error (4 hidden units)
class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None
        self.wo = None


    def learn(self, Xtrain, ytrain):

        std = 1.0 / np.sqrt(Xtrain.shape[1])
        self.numfeatures = Xtrain.shape[1]
        self.wi = std * np.random.randn(self.params['nh'], self.numfeatures)
        std = 1.0 / np.sqrt(self.params['nh'])
        self.wo = std * np.random.randn(1, self.params['nh'])
        # print(self.wi.shape, self.wo.shape)

        stepsize = 0.01
        epochs = self.params['epochs']
        for i in range(epochs):  # (epochs):
            arr = np.arange(Xtrain.shape[0])
            np.random.shuffle(arr)
            for j in arr:
                gradient_1, gradient_2 = self.update(Xtrain[j], ytrain[j]) #compute gradiant using  (back propagation) update
                self.wo = self.wo - stepsize * gradient_2
                self.wi = self.wi - stepsize * gradient_1

        # print(self.wo, self.wi)

    def predict(self,Xtest):
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        for i in range(Xtest.shape[0]):
            h, y = self.evaluate(Xtest[i])
            if y >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0

        assert len(ytest) == Xtest.shape[0]
        return ytest


    def evaluate(self, inputs):
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs.T))

        # output activations
        ao = self.transfer(np.dot(self.wo,ah)).T

        return (
            ah,
            ao,
        )

    def update(self, inputs, outputs):

        h, y_hat = self.evaluate(inputs)
        # print(h.shape, y_hat.shape)

        d_1 = y_hat - outputs
        d_2 = np.zeros(self.params['nh'])

        #the gradients for the cost function with respect to self.w_input and self.w_output.
        grad_output = np.zeros((1, self.params['nh']))
        grad_input = np.zeros((self.params['nh'], self.numfeatures))
        for i in range(self.params['nh']):
            grad_output[0][i] = d_1 * h[i]

        d_2 = np.dot(self.wo.T, d_1)
        for j in range(self.params['nh']):
            # print (h.shape, self.wo.shape)

            d_2[j] = d_2[j] * h[j] * (1 - h[j])
        grad_input = np.outer(d_2, inputs)


        assert grad_input.shape == self.wi.shape
        assert grad_output.shape == self.wo.shape
        return (grad_input, grad_output)


'''
2. Kernel Logistic Regression
'''
# Note: high variance in errors! Make sure to run multiple times
# Susy: ~28 error (40 centers)
class KernelLogisticRegression(LogisticReg):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
            'kernel': 'None'
        }, parameters)
        self.weights = None

    '''2. (a)linear karnel'''
    def linear(self, x, c):
        phi = 0
        #print(x.shape, c.shape)
        for i in range (x.shape[0]):
            phi = phi + x[i]*c[i]
        return phi

    '''2. (b)hamming distence karnel'''
    def hamming(self, x, c):
        phi = 0
        for i in range (len(x)):
            if x[i] != c[i]:
                phi = phi + 1
        return phi


    def Kerneltransform(self, Xtrain):

        #transform to kernel

        phi_train = np.zeros((Xtrain.shape[0], self.numcenters))
        #print(Xtrain.shape, self.centers.shape)

        for i in range (Xtrain.shape[0]):
            for j in range (self.numcenters):
                if self.params['kernel'] == 'linear':
                    phi_train[i][j] = self.linear(Xtrain[i], self.centers[j])
                elif self.params['kernel'] == 'hamming':
                    phi_train[i][j] == self.hamming(Xtrain[i], self.centers[j])
        #print(phi_train.shape)
        return phi_train

    def learn(self, X, y):
        #kernel of input
        phi = None

        self.numcenters = self.params['centers']
        shuffeledX = X
        np.random.shuffle(shuffeledX)
        #print(shuffeledX.shape)
        self.centers = shuffeledX[:self.numcenters]
        #print(self.centers[0])
        phi = self.Kerneltransform(X)
        self.weights = np.zeros(phi.shape[1],)

        epochs = self.params['epochs']
        stepsize = self.params['stepsize']
        numsamples = X.shape[0]
        for i in range (epochs):
            # shuffle data points from 1, ..., numsamples
            arr = np.arange(numsamples)
            np.random.shuffle(arr)
            for j in arr:
                cost_grad = utils.sigmoid(np.dot(phi[j], self.weights)) - y[j]  #cross entropy loss
                #print ('test',cost_grad.shape, phi[j].shape)
                gradient = np.dot(cost_grad, phi[j])
                # print (gradient)
                self.weights = self.weights-stepsize*gradient


    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0], dtype=int)
        p_1 = np.zeros(Xtest.shape[0], dtype=int)

        phi_test = self.Kerneltransform(Xtest) #transfer test input to kernel
        p_1 = utils.sigmoid(np.dot(phi_test, self.weights))
        #print(phi_test.shape)
        #print(self.weights)
        for i in range (len(ytest)):
            if p_1[i] >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0

        return ytest


'''
Bonus (a) Neural Network with 2 hidden layers and ADAM update
'''
# NN with 2 hidden layers
class NeuralNet2(Classifier):
    def __init__(self, parameters={}):
        #super().__init__(parameters)
        #nh1 number of first hidden layer, nh2 number of second hidden layer
        self.params = utils.update_dictionary_items({
            'nh1': 4, 'nh2':8,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None
        self.wo = None
        self.wh = None


    def learn(self, Xtrain, ytrain):

        std = 1.0 / np.sqrt(Xtrain.shape[1])
        self.numfeatures = Xtrain.shape[1]
        self.wi = std * np.random.randn(self.params['nh1'], self.numfeatures)
        std = 1.0 / np.sqrt(self.params['nh1'])
        self.wh = std * np.random.randn(self.params['nh2'], self.params['nh1'])
        std = 1.0 / np.sqrt(self.params['nh2'])
        self.wo = std * np.random.randn(1, self.params['nh2'])


        epochs = self.params['epochs']

        #learn weights
        for i in range(epochs):  # (epochs):
            arr = np.arange(Xtrain.shape[0])

            np.random.shuffle(arr)
            for j in arr:

                gradient_1, gradient_2, gradient_3 = self.update(Xtrain[j], ytrain[j])

                self.wo = self.ADAM(self.wo, gradient_3, i+1)#self.wo - stepsize * gradient_3
                self.wh = self.ADAM(self.wh, gradient_2, i+1)
                self.wi = self.ADAM(self.wi, gradient_1, i+1)
                # print((self.feedforward(Xtrain[j])[1] - ytrain[j]) ** 2)

        # print(self.w_output, self.w_input)

    '''
    ADAM update of waight according to the implementation of Assignment 2
    '''
    def ADAM(self, weights, g_t, t):

        #ADAM parameters
        beta_1 = 0.9
        beta_2 = 0.999
        m_t = np.zeros(g_t.shape)
        v_t = np.zeros(g_t.shape)
        epsilon = 1e-8
        alpha = 0.01

        m_t = beta_1* m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
        v_t = beta_2* v_t + (1-beta_2)*np.square(g_t) 	#updates the moving averages of the squared gradient (element wise sueare of gradient)

        m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates

        weight_prev = weights
        weights = weight_prev - (alpha*m_cap)/(np.sqrt(v_cap)+ epsilon)	#updates the parameters

        return weights


    def predict(self,Xtest):
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        for i in range(Xtest.shape[0]):
            h1, h2, y = self.evaluate(Xtest[i])
            if y >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0

        assert len(ytest) == Xtest.shape[0]
        return ytest


    def evaluate(self, inputs):
        # in activations
        ah1 = self.transfer(np.dot(self.wi,inputs.T))

        # hidden activations
        ah2 = self.transfer(np.dot(self.wh, ah1).T)

        # output activations
        ao = self.transfer(np.dot(self.wo,ah2)).T

        #print('output shape',ao.shape)
        #print('hiden2 shape', ah2.shape)
        #print('hiden1 shape', ah1.shape)

        return (
            ah1,
            ah2,
            ao,
        )



    def update(self, inputs, outputs):

        h1,h2, y_hat = self.evaluate(inputs)
        # print(h.shape, y_hat.shape)
        # print("-----")
        # print (self.feedforward(x))
        # print("-----")
        d_1 = y_hat - outputs
        d_2 = np.zeros(self.params['nh2'])
        d_3 = np.zeros(self.params['nh1'])

        #the gradients for the cost function with respect to self.w_input and self.w_output.
        grad_output = np.zeros((1, self.params['nh2']))
        grad_hiden = np.zeros((self.params['nh2'], self.params['nh1']))
        grad_input = np.zeros((self.params['nh1'], self.numfeatures))
        for i in range(self.params['nh2']):
            grad_output[0][i] = d_1 * h2[i] #d_1.shape 1, h_2.shape(nh2,)

        d_2 = np.dot(self.wo.T, d_1)
        for i in range(self.params['nh2']):
            d_2[i] = d_2[i] * h2[i] * (1 - h2[i]) #d_2.shape nh2, h_2.shape(nh2,), wo.shape(1,nh2)
        grad_hiden = np.outer(d_2, h1)


        d_3 = np.dot(self.wh.T,d_2) #d_2.shape (nh2,), h_2.shape(nh2,), wh.shape(nh2,nh1)
        for j in range(self.params['nh1']):
            d_3[j] = d_3[j] * h1[j] * (1 - h1[j])
        grad_input = np.outer(d_3, inputs)


        assert grad_input.shape == self.wi.shape
        assert grad_output.shape == self.wo.shape
        return (grad_input, grad_hiden, grad_output)
