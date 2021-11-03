Name: Rejwana Haque


Question 1: 
(a)Answered in Scanned Assignment2.pdf
(b)Answered in Scanned Assignment2.pdf
(c)Answered in Scanned Assignment2.pdf

Question 2:

Python version: python 2.7

(a) Class Implementation in regressionalgorithms.py and sending 385 features from script_regression.py

Findings:

When the number of features grows, there is a greater chance of a feature value to be correlated. This makes the singular values to be zero which doesn’t let X to be full rank. This causes Xtranspose*X to be non-invertable. The solution then becomes unstable.
To fix the issue, we can either change the inverse function "np.linalg.inv" to "np.linalg.pinv", which calculate the pseudo-inverse instead, or we can use regularizer.

(b) Implemented in script_regression.py

(c) Implemented in regressionalgorithms.py 

Discussion about result of implemented Ridge Reggression:
Do not need to use psudoinverse as regularization parameter has the effect of shifting the squared singular value removing the stability issue of the Q2(a).
Ridge Regression in (c) will be better in terms of Error since it includes a l2 regularizer which balance the error and prior, or training data and test data, hence works better on new data, i.e. less error on test. 

(d)Lasso regression Implemented in regressionalgorithms.py 

(e)Stochastic Gredient Decent Implemented in regressionalgorithms.py 

Report of Error is ploted in the class SDG in regressionalgorithms.py and also given in Error_SGD in images folder.

(f) For stochastic gradient descent, it is much faster in terms of number of Epoch (less number of time the entire training set is processed) to approach the minima(time to converge) although it is not always heading to the best direction of minima.
For batch gradient descent, it takes longer (more number of times entire training set being processed) to converge, and the error is larger at the beginning. But it always decreases in  the direction of minima and guarantees to converge at some point.

The report of Error vs Epoch is given in Error_vs_Epoch in the images folder.
The report of Error vs Runtime is given in Error_vs_Runtime in the images folder.
Both graphs are also plotted in script_regression.py.


***
Bonus Questions
(a) Update exponential moving average over gradient Implemented in regressionalgorithms.py

(b) ADAM Implemented in regressionalgorithms.py

(c) Update momentum removing initialization bias Implemented in regressionalgorithms.py

Second part of the Question is shown in scanned Assignment2.pdf


