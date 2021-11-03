CMPUT 566 Assignment 3
Rejwana Haque
1627541

Python version: 3.7

Question 1:
 
(a)Including column of ones and not include column of ones has the same result. This is because in Naive Bayes classifier, the presence of a particular feature in a class is independent of the presence of any other feature. And we predict y assuming the features are independent of each other.

In Naive Bayes assumption: 
P(c|x) = P(x[1]|c)*P(x[2]|c) ……* P(x[9]|c)*P(c)

Here P(x[9]|c) = 1 as this corresponds to column of ones (means the probability distribution of the feature is always one for any class level with mean =1 standard deviation = 0 ). This leads to 

P(c|x) = P(x[1]|c)*P(x[2]|c) ……* P(x[9]|c)*P(c)
	= P(x[1]|c)* ……* P(x[8]|c)* 1 *P(c) 

so the column of one does not impact the learning to Naive Bayes.

(b) Implemented in classalgorithms.py.

(c) Implemented in classalgorithms.py.

(d) Implemented in script_classify.py. Errors are also reported as aveerror and stderror in script_classify.py.



Average error for Random: 49.34 
Standerd error for Random: 49.34 +- 2.24693341989e-15
Best parameters for Naive Bayes: {'usecolumnones': False}
Average error for Naive Bayes: 25.78 
Standerd error for Naive Bayes: 25.78 +- 0.0
Best parameters for Naive Bayes Ones: {'usecolumnones': True}
Average error for Naive Bayes Ones: 25.78
Standerd error for Naive Bayes Ones: 25.78 +- 0.0
Best parameters for Linear Regression: {'regwgt': 0.0}
Average error for Linear Regression: 25.02
Standerd error for Linear Regression: 25.02 +- 1.12346670994e-15
Best parameters for Neural Network: {'nh': 4, 'transfer': 'sigmoid', 'epochs': 100, 'stepsize': 0.01}
Average error for Neural Network: 23.08
Standerd error for Neural Network: 23.08 +- 1.12346670994e-15
Best parameters for Logistic Regression: {'regwgt': 0.0, 'regularizer': 'None'}
Average error for Logistic Regression: 26.64
Standerd error for Logistic Regression: 26.64 +- 2.24693341989e-15

Generally, the neural network with 4 hidden units performers better than Linear Regression, than Naive Bayes, than logistic regression. 

 Question 2:

 (a) Kernel logistic regression with a linear kernel has higher error than Naive Bayes, Logistic regression and Neural network. 

 (b) Average error for Random: 49.698  and Average error for Kernel Logistic Regress hamming: 24.188 
Kernel Logistic Regression with Hamming distance kernal outperforms the random predictor significantly.



  Bonus:

(a) Implemented in classalgorithms.py.

(b) The validation error in K-fold CV has higher variance than stratified K-fold CV. Stratified K-fold also estimates test error better than K-fold is the class ratio on the test dataset is same as the train dataset. 



Note: I commented out all learners except Random and Kernel Logistic Regression for answering 2(b). Also set:
parser.add_argument('--dataset', type=str, default="census",help='Specify the name of the dataset')
