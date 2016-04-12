"""
Implement a feedforward neural network using theano
"""
import numpy as np
import theano
import theano.tensor as T

class Layer(object):
    """ A hidden layer in a traditional (feedforward) neural network.
    In common notation, take in a batch of a_(i-1)'s (the output
    of the previous layer) and return a batch of a_i's.


    Params:
        n_inputs -- int, number of nodes in the previous layer
        n_nodes -- int, number of nodes in the current layer
        inputs -- Theano.dmatrix, stores the inputs so that FFNN
                      can string all the layers together.  In common
                      notation, these are a_(i-1) for the batch.
        layer -- int, which layer this is. For theano naming purposes.
        noise_scale -- float, scaling parameter for the amount of
                           initialized noise for weights.  Larger
                           values correspond to more noise.
        nonlin -- type 'theano.tensor.elemwise.Elemwise' or 'function'.
                      This is the choice of nonlinearity for the output
                      of this layer, default is ReLU. Other common
                      options include:
                      theano.tensor.tanh
                      theano.tensor.nnet.sigmoid
                      T.nnet.softmax (for output layer)
    """
    def __init__(self, n_inputs, n_node, inputs, layer,
                 noise_scale, nonlin=T.nnet.relu):
        noise = noise_scale/np.sqrt(n_inputs*n_nodes)
        self.W = theano.shared(noise*np.random.randn(n_inputs,n_nodes),
                               name='W{}'.format(layer))
        self.b = theano.shared(np.zeros(n_nodes),
                               name='b{}'.format(layer))

        self.output = nonlin(X.dot(self.W)+self.b)


class FFNN(object):
    """ A traditional (feedforward) neural network, with a variable
    number of hidden layers.

    Params:
        n_inputs -- int, number of nodes in the input layer
        n_outputs -- int, number of nodes in the output layer
        n_hidden -- array of ints, number of nodes in each hidden layer
        batch -- int the size of the minibatch used in SGD.  The default
                     s 0, corresponding to standard GD.
        reg -- float, the regularization parameter
        alpha -- float, the learning rate
        noise_scale -- float, scaling parameter for the amount of
                           initialized noise for weights.  Larger
                           values correspond to more noise.
        n_epochs -- int, number of training epochs, that is, iterations
                        through the entire training set.
        print_every -- int, print the error after this many epochs
                           while training. Set to 0 to turn off.
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, **kwargs):
        property_defaults = {
            'n_epochs': 100,
            'print_every': 10,
            'reg': .01,
            'alpha': .01,
            'batch': 0
            'noise_scale': 1.0
        }
        for (prop, default) in property_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        self.arch = [n_input] + n_hidden + [n_outputs]
        final = len(self.arch) - 1  # the true "number of layers"

        X = T.dmatrix('X')
        y = T.dmatrix('y') # one-hot outputs

        # Construct layers
        layer_outputs,parameters,weights = [X],[],[]
        for index, layer in enumerate(n_hidden+[n_outputs]):
            nonlin = T.nnet.softmax if index == final-1 else T.nnet.relu
            layer = Layer(n_inputs=self.arch[index],
                          n_nodes=self.arch[index+1],
                          inputs=layers[index][0],
                          layer=index+1,
                          noise_scale=noise_scale,
                          nonlin=nonlin)
            layer_outputs.append(layer.output)
            parameters.extend([layer.W,layer.b])
            weights.append(layer.W)

        # Expressions for building theano functions
        output = layer_outputs[-1]
        prediction = np.argmax(output,axis=1)
        crossentropy = T.nnet.categorical_crossentropy(output,y).mean()
        regularization = reg * sum([(W**2).sum() for W in weights])
        cost = crossentropy + regularization

        # gradients
        grads = T.grad(cost,parameters)
        updates = [p,p - self.alpha*g for p,g in zip(parameters,grads]

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

        I = np.identity(self.n_outputs)
        preds = np.array([I[i] for i in self.predict(X_data)])
        wrong = (preds != y_data).sum() / 2

        score = (n_samples*1.0 - total_wrong)/n_samples
        return total_wrong, score
