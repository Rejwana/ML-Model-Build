import numpy as np

import MLCourse.dataloader as dtl
import MLCourse.utilities as utils
import classalgorithms as algs
import math as mt

from random import randrange

def getaccuracy(ytest, predictions):
    correct = 0
    # count number of correct predictions
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    #correct = np.sum(ytest == predictions)
    # return percent correct
    return (correct / float(len(ytest))) * 100

def geterror(ytest, predictions):
    return (100 - getaccuracy(ytest, predictions))

""" k-fold cross-validation
K - number of folds
X - data to partition
Y - targets to partition
Algorithm - the algorithm class to instantiate
parameters - a list of parameter dictionaries to test

NOTE: utils.leaveOneOut will likely be useful for this problem.
Check utilities.py for example usage.
"""
'''
Bonus (b) Stratified K-fold Cross Validation
'''
def stratified_CV (K, X, Y, Algorithm, parameters):
    all_errors = np.zeros((len(parameters), K))
    arr = np.arange(K)
    #count of class 0 and class 1
    C_0 = 0
    C_1 = 1
    #hold the data points according to class lebel
    Y_0 = np.empty([0, ])
    Y_1 = np.empty([0, ])
    X_0 = np.empty(shape =[0,X.shape[1]])
    X_1 = np.empty(shape = [0,X.shape[1]])

    #print(X.shape, Y.shape, Y_0.shape, X_0.shape)

    #X_train = np.empty(shape=[0, X.shape[1]])
    #Y_train = np.empty([0, ])



    #seperating samples into two groups according to class lebel
    for i in range(len(Y)):
        #print(X[i].shape, X_0.shape)
        X_Train = np.reshape(X[i], (1, X.shape[1]))
        if Y[i] == 0.0:
            C_0 += 1
            Y_0 = np.append(Y_0, Y[i])
            X_0 = np.concatenate([X_0, X_Train])
        else:
            C_1 += 1
            Y_1 = np.append(Y_1, Y[i])
            X_1 = np.concatenate([X_1, X_Train])


    #spliting each class lebeled dataset into K folds
    X0_fold = np.array_split(X_0, K)
    Y0_fold = np.array_split(Y_0, K)
    X1_fold = np.array_split(X_1, K)
    Y1_fold = np.array_split(Y_1, K)


    for k in range(K):

        trainfolds = utils.leaveOneOut(arr, k)
        #X_validate and Y_validate hold the validation fold of both class lebel
        X_validate = np.empty(shape=[0, X.shape[1]])
        Y_validate = np.empty([0, ])

        for i, params in enumerate(parameters):
            Larner = Algorithm(params)
            X_train = np.empty(shape=[0, X.shape[1]])
            Y_train = np.empty([0, ])
            #take one fold of class 0 and one fold of class 1
            X_validate = np.concatenate([X_validate, X0_fold[k]])
            Y_validate = np.concatenate([Y_validate, Y0_fold[k]])
            X_validate = np.concatenate([X_validate, X1_fold[k]])
            Y_validate = np.concatenate([Y_validate, Y1_fold[k]])

            # concatenation of k-1 folds
            for j in trainfolds:
                #take remaining folds of both class label as train set
                X_train = np.concatenate([X_train, X0_fold[j]])
                Y_train = np.concatenate([Y_train, Y0_fold[j]])
                X_train = np.concatenate([X_train, X1_fold[j]])
                Y_train = np.concatenate([Y_train, Y1_fold[j]])
                #print(X_train.shape, Y_train.shape)
            #train on train set
            Larner.learn(X_train, Y_train)  # learning on remaining folds other than k
            #predict on validation set
            predictions = Larner.predict(X_validate)
            all_errors[i][k] = geterror(Y_validate, predictions)
            print('error for ' + str(params) + ' on cv fold:' + str(k) + ': ' + str(all_errors[i][k]))

    avg_errors = np.mean(all_errors, axis=1)
    best_parameters = parameters[0]
    best_error = avg_errors[0]
    for i, params in enumerate(parameters):
        avg_errors[i] = np.mean(all_errors[i])
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])
        if avg_errors[i] < best_error:
            best_error = avg_errors[i]
            best_parameters = params

    print('Best Parameter', best_parameters)
    return best_parameters


'''
1 (d) K-fold  Cross Validation
'''

def cross_validate(K, X, Y, Algorithm, parameters):
    all_errors = np.zeros((len(parameters), K))
    arr = np.arange(K)
    #print(X.shape, Y.shape)

    X_fold = np.split(X, K)
    Y_fold = np.split(Y, K)


    for k in range(K):

        trainfolds = utils.leaveOneOut(arr, k)
        for i, params in enumerate(parameters):
            Larner = Algorithm(params)
            X_train = np.empty(shape = [0,X.shape[1]])
            Y_train = np.empty([0,])

            #concatenation of k-1 folds
            for j in trainfolds:
                X_train = np.concatenate([X_train,X_fold[j]])
                Y_train = np.concatenate([Y_train,Y_fold[j]])
                print(X_train.shape, Y_train.shape)

            Larner.learn(X_train, Y_train) # learning on remaining folds other than k
            predictions = Larner.predict(X_fold[k])
            all_errors[i][k] = geterror(Y_fold[k], predictions)
            print('error for ' + str(params) + ' on cv fold:' + str(k) + ': ' + str(all_errors[i][k]))

    avg_errors = np.mean(all_errors, axis=1)
    best_parameters = parameters[0]
    best_error = avg_errors[0]
    for i, params in enumerate(parameters):
        avg_errors[i] = np.mean(all_errors[i])
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])
        if avg_errors[i] < best_error:
            best_error = avg_errors[i]
            best_parameters = params

    print('Best Parameter',best_parameters)
    return best_parameters

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--trainsize', type=int, default=5000,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=5000,
                        help='Specify the test set size')
    parser.add_argument('--numruns', type=int, default=10,
                        help='Specify the number of runs')
    # for census dataset commentout this statement
    parser.add_argument('--dataset', type=str, default="susy",
                        help='Specify the name of the dataset')

    #parser.add_argument('--dataset', type=str, default="census",
                        #help='Specify the name of the dataset')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    numruns = args.numruns
    dataset = args.dataset



    classalgs = {
        'Random': algs.Classifier,
        'Naive Bayes': algs.NaiveBayes,
        'Linear Regression': algs.LinearRegressionClass,
        'Logistic Regression': algs.LogisticReg,
        'Neural Network': algs.NeuralNet,
        'Neural Network h2': algs.NeuralNet2,
        'Kernel Logistic Regression': algs.KernelLogisticRegression,
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        # name of the algorithm to run
        'Naive Bayes': [
            # first set of parameters to try
            { 'usecolumnones': True },
            # second set of parameters to try
            { 'usecolumnones': False },
        ],
        'Logistic Regression': [
            { 'stepsize': 0.001 },
            { 'stepsize': 0.005 },
            { 'stepsize': 0.01 },
        ],
        'Neural Network h2': [
            { 'epochs': 100, 'nh1': 4, 'nh2': 4 },
            { 'epochs': 100, 'nh1': 8, 'nh2': 4 },
            { 'epochs': 100, 'nh1': 4, 'nh2': 8 },
            { 'epochs': 100, 'nh1': 8, 'nh2': 8 },
        ],
        'Neural Network': [
            {'epochs': 100, 'nh': 4},
            { 'epochs': 100, 'nh': 8 },
            { 'epochs': 100, 'nh': 16 },
            { 'epochs': 100, 'nh': 32 },
        ],

        'Kernel Logistic Regression': [
            { 'centers': 10, 'stepsize': 0.01, 'kernel': 'linear' },
            { 'centers': 20, 'stepsize': 0.01, 'kernel': 'linear' },
            { 'centers': 40, 'stepsize': 0.01, 'kernel': 'linear' },
            { 'centers': 80, 'stepsize': 0.01, 'kernel': 'linear'},
            #for census dataset
            #{'centers': 10, 'stepsize': 0.01, 'kernel': 'hamming'},
            #{ 'centers': 20, 'stepsize': 0.01, 'kernel': 'hamming' },
            #{ 'centers': 40, 'stepsize': 0.01, 'kernel': 'hamming'},
            #{ 'centers': 80, 'stepsize': 0.01, 'kernel': 'hamming' },
        ]
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros(numruns)

    for r in range(numruns):
        if dataset == "susy":
            trainset, testset = dtl.load_susy(trainsize, testsize)
        elif dataset == "census":
            trainset, testset = dtl.load_census(trainsize,testsize)
        else:
            raise ValueError("dataset %s unknown" % dataset)

        # print(trainset[0])
        Xtrain = trainset[0]
        Ytrain = trainset[1]
        # cast the Y vector as a matrix
        #Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])
        #print(Xtrain.shape)

        Xtest = testset[0]
        Ytest = testset[1]
        # cast the Y vector as a matrix
        Ytest = np.reshape(Ytest, [len(Ytest), 1])

        best_parameters1 = {}
        best_parameters2 = {}
        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [ None ])

            #print(Xtrain.shape, Ytrain.shape)

            '''
            For comparing k-fold CV and stratified k-fold CV
            '''

            best_parameters1[learnername] = cross_validate(5, Xtrain, Ytrain, Learner, params)
            best_parameters2[learnername] = stratified_CV(5, Xtrain, Ytrain, Learner, params)


        for learnername, Learner in classalgs.items():
            params = best_parameters1[learnername]
            #params = best_parameters2[learnername]
            learner = Learner(params)
            learner.learn(Xtrain, Ytrain)
            predictions = learner.predict(testset[0])
            errors[learnername][r] = geterror(testset[1], predictions)
    print('error', errors)

    '''
    1(d) Avarage and Standard Error
    '''

    for learnername in classalgs:
        aveerror = np.mean(errors[learnername])
        print('Average error for ' + learnername + ': ' + str(aveerror))

        stderror = np.std(errors[learnername])/ mt.sqrt(numruns)
        print('Standered error for ' + learnername + ': ' + ':' + str(aveerror) + ' +- ' + str(stderror))

