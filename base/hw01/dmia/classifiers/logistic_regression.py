#i'm using python3
import numpy as np
from scipy import sparse
from scipy.special import expit


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this classifier using stochastic gradient descent.

        Inputs:
        - X: N x D array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # Add a column of ones to X for the bias sake.
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in range(num_iters):
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            X_batch = None
            y_batch = None            
            rand_idx = np.random.choice(num_train, batch_size)
            X_batch = X[rand_idx, :]
            y_batch = y[rand_idx]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w += -1 * learning_rate * gradW

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return self

    def predict_proba(self, X, append_bias=False):
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of data. Each row is a D-dimensional point.
        - append_bias: bool. Whether to append bias before predicting or not.

        Returns:
        - y_proba: Probabilities of classes for the data in X. y_pred is a 2-dimensional
          array with a shape (N, 2), and each row is a distribution of classes [prob_class_0, prob_class_1].
        """
        if append_bias:
            X = LogisticRegression.append_biases(X)
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the probabilities of classes in y_proba.   #
        # Hint: It might be helpful to use np.vstack and np.sum                   #
        ###########################################################################
        y_proba = np.vstack((expit(X.dot(self.w)), 1 - expit(X.dot(self.w))))



        ###########################################################################
        #                           END OF Your CODE                              #
        ###########################################################################
        return y_proba

    def predict(self, X):
        """
        Use the ```predict_proba``` method to predict labels for data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        y_proba = self.predict_proba(X, append_bias=True)
        y_pred = [1 if y_proba[0][i] >= .5 else -1 for i in range(len(y_proba[0]))]

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """Logistic Regression loss function
        Inputs:
        - X: N x D array of data. Data are D-dimensional rows
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes
        Returns:
        a tuple of:
        - loss as single float
        - gradient with respect to weights w; an array of same shape as w
        """
        dw = np.zeros_like(self.w)  # initialize the gradient as zero
        loss = 0
        # Compute loss and gradient. Your code should not contain python loops.
        sig = (1 + np.exp(-X_batch.dot(self.w) * y_batch))
        loss =  np.sum(np.log(sig))
        dw = X_batch.T.dot((1./sig - 1) *  y_batch)
        num_train = X_batch.shape[0]
        
        loss /= num_train
        dw /= num_train

        loss += .5 * reg * np.dot(self.w[:-1], self.w[:-1])
        dw[:-1] += reg * self.w[:-1]

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        # Note that the same thing must be done with gradient.


        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.


        return loss, dw

    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
