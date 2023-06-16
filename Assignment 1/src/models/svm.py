"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        #D is number of features in the data
        #D = 3072
        #self.w = np.zeros((D, n_class)) 
        self.w = np.random.randn(3072, n_class) / np.sqrt(3072)
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
    
        D = self.w.shape[0]
        N = X_train.shape[0]

        ''' initialize the array that hold the gradient with zeros and this array has
        the same shape as w
        '''
        gradients = np.zeros((D, self.n_class))

        for i in range(N):
            #compute predicted scores
            predictions = X_train[i].dot(self.w)

            # true score 
            true_y = predictions[y_train[i]]

            # calculate loss at the j-th position in the predictions
            for j in range(self.n_class):
                # if correct do nothing
                if j == y_train[i]:
                    continue
                    
                # delta=1
                m = predictions[j] - true_y + 1

                if m > 0:
                    gradients[:, j] += X_train[i]
                    gradients[:, y_train[i]] -= X_train[i]




        # take the average 
        gradients /= N
        
        # add regularization
        gradients += self.reg_const * self.w


        return gradients


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        
        for epoch in range(self.epochs):
            for i in range(0, len(y_train), 128):
                X_mini_batch = X_train[i:i + 128]
                y_mini_batch = y_train[i:i + 128]
                
                
                # to add extra 1 for matching bias b 
                # arr = np.ones((3072,))
                # X_mini_batch = np.append(arr, 1)
                # y_mini_batch = X_mini_batch.reshape(1, -1)
                # print(X_mini_batch.shape)
                # print(self.w.shape)
                
                # apply lr decay using Exponenial decay
                # self.alpha *= np.exp(-0.9 * i) 
                
                # #apply lr decay unsing 
                # self.alpha /= (1 + 0.5 * i)
                
                gradient = self.calc_gradient(X_mini_batch, y_mini_batch)
                # SGD update w = w - lr * gradient
                self.w -= self.alpha * gradient
                

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
        # to add extra 1 for matching bias b 
        # arr = np.ones((3072,))
        # X_test = np.append(arr, 1)
        # X_test = X_test.reshape(1, -1)
        scores = np.dot(X_test, self.w)
        predictions = np.argmax(scores, axis=1)
        return predictions
