"""
Implement a feedforward neural network using theano
"""
import numpy as np
import theano
import theano.tensor as T

class FFNN(object):
    """ A traditional (feedforward) neural network, with a variable number
    of hidden layers.

    Params:
        n_inputs -- int, number of nodes in the input layer
        n_outputs -- int, number of nodes in the output layer
        n_hidden -- array of ints, number of nodes in each hidden layer
        batch -- int the size of the minibatch used in SGD.  The default is
                    0, corresponding to standard GD.
        reg -- float, the regularization parameter
        alpha -- float, the learning rate
        n_epochs -- int, number of training epochs, that is, iterations
                        through the entire training set.
        print_every -- int, print the error after this many epochs
                           while training. Set to 0 to turn off.
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, **kwargs):
        property_defaults = {
            'epochs': 1000,
            'print_every': 100,
            'reg': .1,
            'alpha': .01,
            'batch': 0
        }
        for (prop, default) in property_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        self.arch = [n_inputs] + n_hidden + [n_outputs]

        X = T.dmatrix('X')
        y = T.dmatrix('y') # one-hot outputs

        # Weights and biases
        # TODO: extend to multiple hidden layers!
        noise = 1/np.sqrt(n_inputs*n_hidden[0]*n_outputs)
        self.W1 = theano.shared(noise*np.random.randn(n_inputs,n_hidden[0]),
                                name='W1')
        self.b1 = theano.shared(np.zeros(n_hidden[0]), name='b1')
        self.W2 = theano.shared(noise*np.random.randn(n_hidden[0],n_outputs),
                                name='W2')
        self.b2 = theano.shared(np.zeros(n_outputs), name='b2')

        # Feedforward
        z1 = X.dot(self.W1)+self.b1
        hidden = T.tanh(z1)
        z2 = hidden.dot(self.W2) + self.b2
        output = T.nnet.softmax(z2)
        prediction = np.argmax(output,axis=1)
        crossentropy = T.nnet.categorical_crossentropy(output,y).mean()
        regularization = self.reg*((self.W1**2).sum()+(self.W2**2).sum())
        cost = crossentropy + regularization

        # gradients
        gW1,gb1,gW2,gb2 = T.grad(cost,[self.W1,self.b1,self.W2,self.b2])
        updates = ((self.W1,self.W1-self.alpha*gW1),
                   (self.b1,self.b1-self.alpha*gb1),
                   (self.W2,self.W2-self.alpha*gW2),
                   (self.b2,self.b2-self.alpha*gb2))

        # build theano functions for gradient descent and model tuning
        self.epoch = theano.function(inputs = [X,y],
                                     outputs = [],
                                     updates = updates)
        self.count_cost = theano.function(inputs = [X,y],outputs = cost)
        self.predict = theano.function(inputs=[X],outputs=prediction)

    def fit(self,X_data,y_data):
        """ Fit the model. Performs standard or batch gradient descent.

        Params:
            X_data -- 2d np.array of training data, rows are samples and
                          columns are features
            y_data -- 2d np.array of one-hot labels

        Return:
            array of (int,float), cost after periodic epochs
        """
        costs = []
        def run_epoch(X_data,y_data):
            self.epoch(X_data,y_data)
            if i % self.print_every == 0:
                costs.append((i,self.count_cost(X_data,y_data)))

        if self.batch == 0:
            # performing vanilla gradient descent
            for i in range(self.epochs):
                run_epoch(X_data,y_data)
        else:
            # performing minibatch gradient descent
            n_samples = len(X_data)
            for i in range(self.epochs):
                for ind in range(0,n_samples,self.batch):
                    rows = list(range(ind,min(ind+self.batch,n_samples)))
                    run_epoch(X_data[rows,:],y_data[rows,:])
                np.random.shuffle(arr) # reduces bias in training
        return costs

    def accuracy(self,X_data,y_data):
        """ Compute model accuracy.  If you send in the same dataset as
        was used to train, will compute training accuracy, otherwise will
        compute validation accuracy.

        Params:
            X_data -- 2d np.array of training data, rows are samples and
                          columns are features
            y_data -- 2d np.array of one-hot labels

        Return:
            (int,float), missclassifications and accuracy (as a decimal)
        """

        n_samples = len(X_data)

        I = np.identity(self.arch[-1])
        preds = np.array([I[i] for i in self.predict(X_data)])
        wrong = (preds != y_data).sum() / 2

        score = (n_samples*1.0 - wrong)/n_samples
        return wrong, score
