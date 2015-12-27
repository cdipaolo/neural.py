import numpy as np
from math import sqrt

class FeedForwardNet():
    '''FeedForwardNet
    class implements a feed forward neural
    network with a variable number of layers
    and nonlinearities
    '''

    def __init__(self, layers, transforms, cost='maxent'):
        '''___init___
        initialze the feed forward neural network

        layers -> []uint - dimensions of layers. First
            element is the number of inputs. Last layer
            is the number of outputs

        transforms -> []NonLinearity - the transforms
            for each of the layers (not including the
            first layer). All NonLinearities must
            have a f(x float) -> float and a
            df(x float) -> float method associated
            with the class.

        lr -> float - learning rate (ie. 0.1)

        iterations -> uint - maximum number of iterations
            for the model (ie. 1000)

        cost -> string - objective function. Default to
            'maxent' for Maximum Entropy; can use 'mse'
            for mean squared error
        '''
        assert len(layers) > 1, "Must have at least an input and output layer"
        assert len(layers) == len(transforms)+1, "Must have one less transform than layers (because you don't include the input layer in transforms)"

        self.max_entropy = True if cost == 'maxent' else False

        # inputs to the first layer
        # AKA dimensionality of feature
        # vectors
        self.inputs = layers[0]
        self.layers = layers

        self.transforms = transforms

        # define activation and deltas
        # of layers to use for forward
        # and backprop
        self.activations = list(map(lambda x: np.zeros(x), layers[1:]))
        self.deltas = list(map(lambda x: np.zeros(x), layers[1:]))

        self.weights = []
        for i in range(1, len(layers)):
            dist = 1 / sqrt(layers[i-1])
            self.weights.append(np.random.rand(layers[i-1]+1, layers[i])*2*dist - dist) # draw from Unif(-dist, dist)

    def _forward(self, x):
        '''_forward
        Run a forward pass of an input through
        the network. Input is a numpy matrix/array
        of input examples.

        >>> import nonlinearity as nl
        >>> nn = FeedForwardNet(layers=[3,2,3,1], transforms=[nl.ReLu, nl.ReLu, nl.Sigmoid])
        >>> nn.weights = [np.array([[0,0],[-3,3],[2,5],[10,2]]), np.array([[-10,3,-5],[-2,3,12],[6,1,2]]), np.array([[-20],[0.001],[1],[0.25]])]
        >>> nn._forward(np.array([[1,2,-1],[-1,3,2]])).T
        array([[ 0.15525053,  1.        ]])
        '''
        # add a column of ones for the bias term
        self.activations[0] = self.transforms[0].f(np.column_stack((np.ones(x.shape[0]), x)).dot(self.weights[0]))

        for i in range(1, len(self.layers)-1):
            # calculate activation
            self.activations[i] = self.transforms[i].f(np.column_stack((np.ones(self.activations[i-1].shape[0]), self.activations[i-1])).dot(self.weights[i]))

        return self.activations[-1]
