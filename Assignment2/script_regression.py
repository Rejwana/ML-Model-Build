from __future__ import division
import csv
import random
import math
import numpy as np

import matplotlib.pyplot as plt

import regressionalgorithms as algs

import MLCourse.dataloader as dtl
import MLCourse.plotfcns as plotfcns



def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction, ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction, ytest), ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction, ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions, ytest) / np.sqrt(ytest.shape[0])
    #MSE
    #return 0.5*l2err_squared(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 5

    regressionalgs = {
        'Random': algs.Regressor,
        'Mean': algs.MeanPredictor,
        'FSLinearRegression': algs.FSLinearRegression,
        'RidgeLinearRegression': algs.RidgeLinearRegression,
        # 'KernelLinearRegression': algs.KernelLinearRegression,
        'LassoRegression': algs.LassoRegression,
        'SGD' : algs.SGD,
        'BatchGD': algs.batchGD,
        'Adam' : algs.ADAM,
        'Adam_WB': algs.ADAM_WithoutBias,
        'Momentum' : algs.Momentum,

        # 'LinearRegression': algs.LinearRegression,
        # 'MPLinearRegression': algs.MPLinearRegression,
    }
    numalgs = len(regressionalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        'FSLinearRegression': [
            { 'features': [1, 2, 3, 4, 5] },
            { 'features': [1, 3, 5, 7, 9] },
            { 'regwgt': 0.0, 'features': range(385)}
        ],
        'RidgeLinearRegression': [
            #{ 'regwgt': 0.00 },
            { 'regwgt': 0.01 },
            { 'regwgt': 0.05 },
        ],

    }

    errors = {}

    x = {}
    y = {}
    time = {}

    for learnername in regressionalgs:
        # get the parameters to try for this learner
        # if none specified, then default to an array of 1 parameter setting: None
        params = parameters.get(learnername, [ None ])
        errors[learnername] = np.zeros((len(params), numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0], r))

        for learnername, Learner in regressionalgs.items():
            params = parameters.get(learnername, [ None ])
            for p in range(len(params)):
                learner = Learner(params[p])
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])

                '''plot for SGD,BGD. comment out for other reggressor'''
                if(learnername=='SGD' or learnername== 'BatchGD'):
                    y[learnername], time[learnername] = learner.data() # fetch error and time for SGD and BGD


                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p, r] = error


    for learnername, Learner in regressionalgs.items():
        params = parameters.get(learnername, [ None ])
        besterror = np.mean(errors[learnername][0, :])
        bestparams = 0
        for p in range(len(params)):
            #average error for all runs with parameter p
            aveerror = np.mean(errors[learnername][p, :])

            """ Question2 b) Standard error  """
            stderror = np.std(errors[learnername][p, :])/ math.sqrt(numruns)
            learner = Learner(params[p])
            print('Standered error for ' + learnername + ': ' + str(learner.getparams()) +':' + str(aveerror) + ' +- ' + str(stderror))

            if aveerror < besterror:
                besterror = aveerror
                bestparams = p
        #Learner.reset(parameters[bestparams])

        #plt.plot(x[learnername], y[learnername], label=learnername)



        # Extract best parameters
        #best = params[bestparams]
        #print ('Best parameters for ' + learnername + ': ' + str(best))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(1.96 * np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns)))

        print ('Standered error for ' + learnername + ': ' + str(besterror) + ' +- ' + str( np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns)))

    """ 2 f)  Draw plot of error versus epoches for SGD and BGD """

    x['SGD'] = np.arange(1000)
    x['BatchGD'] = np.arange(len(y['BatchGD']))
    plt.plot(x['SGD'], y['SGD'], label='StochasticGD')
    plt.plot(x['BatchGD'], y['BatchGD'], label='BatchGD')
    plt.xlabel('Number of Epoches')
    plt.ylabel('Error')
    plt.title('Error VS Epoches for BGD and SGD')
    x_limit = max(len(y['SGD']),len(y['BatchGD']))
    plt.xlim([0,x_limit])
    plt.ylim([0,20])
    plt.legend()
    plt.show()

    """ 2 f) Draw plot of error versus time for SGD and BGD """
    plt.plot(time['SGD'], y['SGD'], label='StochasticGD')
    plt.plot(time['BatchGD'], y['BatchGD'], label='BatchGD')
    plt.xlabel('Runtime')
    plt.ylabel('Error')
    plt.title('Error VS Runtime for BGD and SGD')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()