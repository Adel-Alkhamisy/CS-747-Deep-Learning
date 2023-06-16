"""Perceptron model."""

import numpy as np


from operator import add, sub
class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        # TODO: change this
        #D is number of features in the data
        D = 3072
        self.w = np.zeros((D, n_class))  
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        for epoch in range(self.epochs):
            # Find prediction Xw matrix
            for i in range(X_train.shape[0]):
                xi = X_train[i]
                yi = y_train[i]
                # Find dot product xi*W
                scores = np.dot(xi, self.w)
                y_pred = np.argmax(scores)
                
                # apply lr decay using Exponenial decay
                # self.lr *= np.exp(-0.5 * i) 
                
                # #apply lr decay unsing 1/t
                # self.lr /= (1 + 0.5 * i)
                
                if y_pred != yi:
                    self.w[:, yi] = self.w[:, yi] + self.lr * xi
                    self.w[:, y_pred] = self.w[:, y_pred] - self.lr * xi



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
        y_pred = np.argmax(np.dot(X_test, self.w), axis=1)
        return y_pred