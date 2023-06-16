"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = self.w = np.random.randn(3072, n_class) / np.sqrt(3072)  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        N, D = X_train.shape
        C = self.n_class

        # compute scores
        scores = X_train.dot(self.w)

        # Normalize the scores 
        scores -= np.max(scores, axis=1, keepdims=True)

        # compute the exponentials of the scores
        exp_of_scores = np.exp(scores)

        # compute the softmax probabilities
        probs = exp_of_scores / np.sum(exp_of_scores, axis=1, keepdims=True)

        # gradient initialization 
        gradient = np.zeros_like(self.w)

        # compute the gradient
        for i in range(N):
            for j in range(C):
                if j == y_train[i]:
                    gradient[:, j] -= (1 - probs[i, j]) * X_train[i]
                else:
                    gradient[:, j] += probs[i, j] * X_train[i]

        # average
        gradient /= N
        #add regulrization
        gradient += self.reg_const * self.w

        return gradient

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N = X_train.shape[0]

        for epoch in range(self.epochs):
            for i in range(0, N, 32):
                X__mini_batch = X_train[i:i + 32]
                y_mini_batch = y_train[i:i + 32]

                grad = self.calc_gradient(X__mini_batch, y_mini_batch)
                
                # apply lr decay using Exponenial decay
                # self.lr *= np.exp(-0.5 * i) 
                
                # #apply lr decay unsing 
                # self.lr /= (1 + 0.5 * i)

                self.w -= self.lr * grad

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        scores = np.dot(X_test, self.w)
        predictions = np.argmax(scores, axis=1)
        return predictions
