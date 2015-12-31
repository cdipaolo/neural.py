import numpy as np
from math import sqrt,log

class FeedForwardNet():
    '''FeedForwardNet
    class implements a feed forward neural
    network with a variable number of layers
    and nonlinearities
    '''

    def __init__(self, layers, transforms, cost='xent'):
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
            'xent' for Cross Entropy; can use 'mse'
            for mean squared error
        '''
        assert len(layers) > 1, "Must have at least an input and output layer"
        assert len(layers) == len(transforms)+1, "Must have one less transform than layers (because you don't include the input layer in transforms)"

        self.x_entropy = True if cost == 'xent' else False

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

        self.biases = []
        self.weights = []
        for i in range(1, len(layers)):
            dist = 1 / sqrt(layers[i-1])
            self.weights.append(np.random.rand(layers[i-1], layers[i])*2*dist - dist) # draw from Unif(-dist, dist)
            self.biases.append(np.random.rand(1, layers[i])*2*dist - dist)

    def _forward(self, x):
        '''_forward
        Run a forward pass of an input through
        the network. Input is a numpy matrix/array
        of input examples.

        >>> import nonlinearity as nl
        >>> nn = FeedForwardNet(layers=[3,2,3,1], transforms=[nl.ReLu, nl.ReLu, nl.Sigmoid])
        >>> nn.weights = [np.array([[-3,3],[2,5],[10,2]]), np.array([[-2,3,12],[6,1,2]]), np.array([[0.001],[1],[0.25]])]
        >>> nn.biases = [np.array([0,0]),np.array([-10,3,-5]),np.array([-20])]
        >>> nn._forward(np.array([[1,2,-1],[-1,3,2]])).T
        array([[ 0.15525053,  1.        ]])
        '''
        # add a column of ones for the bias term
        self.activations[0] = self.transforms[0].f(x.dot(self.weights[0]) + self.biases[0])

        for i in range(1, len(self.layers)-1):
            # calculate activation
            self.activations[i] = self.transforms[i].f(self.activations[i-1].dot(self.weights[i]) + self.biases[i])

        return self.activations[-1]

    def _backward(self, x, y):
        '''_backward
        Run a backward pass through the network,
        returning the a list of numpy matrices
        of the derivative of the cost function
        with respect to each layer's weights.

        The function assumes that the forward
        pass result is stored in self.activations[-1]

        >>> import nonlinearity as nl
        >>> nn = FeedForwardNet(layers=[3,2,3,1], transforms=[nl.ReLu, nl.ReLu, nl.Sigmoid], cost='mse')
        >>> nn.weights = [np.array([[-3,3],[2,5],[10,2]]), np.array([[-2,3,12],[6,1,2]]), np.array([[0.001],[1],[0.25]])]
        >>> nn.biases = [np.array([0,0]),np.array([-10,3,-5]),np.array([-20])]
        >>> x = np.array([[1,2,-1]])
        >>> y_hat = nn._forward(x) # compute forward pass
        >>> grad_w, grad_b = nn._backward(x, 1)
        >>> print(grad_w)
	[array([[ 0.        , -0.16684527],
	       [ 0.        , -0.33369055],
	       [ 0.        ,  0.16684527]]), array([[ 0.        ,  0.        ,  0.        ],
	       [-0.00121866, -1.21865738, -0.30466435]]), array([[-6.20407395],
	       [-1.55101849],
	       [-1.88337959]])]
        >>> print(grad_b)
        [array([[ 0.        , -0.16684527]]), array([[-0.00011079, -0.11078703, -0.02769676]]), array([[-0.11078703]])]
        '''
        # compute deltas (starting with last layer)
        if self.x_entropy: # assume using sigmoid or softmax
            self.deltas[-1] = (self.activations[-1] - y)
        else:
            self.deltas[-1] = (self.activations[-1] - y) * self.transforms[-1].df(self.activations[-1])

        # iterate through the layers backwards
        for i in range(2, len(self.layers)):
            self.deltas[-i] = np.dot(self.deltas[-i+1], self.weights[-i+1].T) * self.transforms[-i].df(self.activations[-i])

        # calculate gradients
        bias_shape = (x.shape[0],1)
        grad_w = [x.T.dot(self.deltas[0])]
        grad_b = [np.ones(bias_shape).T.dot(self.deltas[0])]
        for i in range(1, len(self.deltas)):
            grad_w.append(self.activations[i-1].T.dot(self.deltas[i]))
            grad_b.append(np.ones(bias_shape).T.dot(self.deltas[i]))
        return grad_w, grad_b

    def cost(self, x, y):
        '''cost
        Cost function computation on the neural
        network. If the cost of the network is
        'x_ent' then use cross entropy. Otherwise
        use mean squared error.

        x is an input array or vector of features
        where each row is another example and each
        column is a feature input.

        y is the label of the result as a matrix/vector
        with just as many rows as x.

        Mean Squared Error Tests:
        >>> import nonlinearity as nl
        >>> nn = FeedForwardNet(layers=[3,2,3,1], transforms=[nl.ReLu, nl.ReLu, nl.Sigmoid], cost='mse')
        >>> nn.weights = [np.array([[-3,3],[2,5],[10,2]]), np.array([[-2,3,12],[6,1,2]]), np.array([[0.001],[1],[0.25]])]
        >>> nn.biases = [np.array([0,0]),np.array([-10,3,-5]),np.array([-20])]
        >>> x = np.array([[1,2,-1],[-1,3,2]])
        >>> nn.cost(x, np.array([[1],[1]]))
        0.17840041878634177
        >>> nn.cost(np.array([1,2,-1]), 1)
        0.35680083757268355
        >>> nn.cost(np.array([-1,3,2]), 1)
        0.0

        Cross Entropy Cost Tests:
        >>> import nonlinearity as nl
        >>> nn = FeedForwardNet(layers=[2,2,3], transforms=[nl.ReLu, nl.Softmax])
        >>> nn.weights = [np.array([[1,3],[2,-1]]), np.array([[1,3,5],[2,4,6]])]
        >>> nn.biases = [np.array([3,-6]),np.array([2,-10,-8])]
        >>> x = np.array([[1,-1],[100,-1]])
        >>> nn._forward(np.array([[1,-1],[100,-1]]))
	array([[  8.80536902e-01,   2.95387223e-04,   1.19167711e-01],
               [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
        >>> nn.cost(x, np.array([[0,0,1],[0,0,1]]))
        2.127223441901406
        >>> nn.cost(np.array([1,-1]), np.array([0,0,1]))
        2.127223441901406
        >>> nn.cost(np.array([100,-1]), np.array([0,0,1]))
	-0.0

        >>> import nonlinearity as nl
        >>> nn = FeedForwardNet(layers=[3,2,3,1], transforms=[nl.ReLu, nl.ReLu, nl.Sigmoid])
        >>> nn.weights = [np.array([[-3,3],[2,5],[10,2]]), np.array([[-2,3,12],[6,1,2]]), np.array([[0.001],[1],[0.25]])]
        >>> nn.biases = [np.array([0,0]),np.array([-10,3,-5]),np.array([-20])]
        >>> x = np.array([[1,2,-1],[-1,3,2]])
        >>> nn.cost(x, np.array([[1],[1]]))
	1.8627151751310125
        >>> nn.cost(np.array([1,2,-1]), 1)
	1.8627151751310125
        >>> nn.cost(np.array([-1,3,2]), 1)
	-0.0
        '''
        # do some casting to make computation
        # efficient (give np axis of {0,1})
        if not hasattr(y, '__len__'):
            y = np.array(y)
            y.shape = (1,1)
        if not hasattr(x, '__len__'):
            x = np.array(x)
            x.shape = (1,1)
        if len(x.shape) == 1:
            x.shape = (1, x.shape[0])
        if len(y.shape) == 1:
            y.shape = (1, y.shape[0])
            
        y_hat = self._forward(x)
        if self.x_entropy:
            if self.layers[-1] == 1: # bernoilli output
                zeros = np.invert(y == 1)
                y_hat[zeros] = 1 - y_hat[zeros]
                return -1*np.log(y_hat).sum()
            # else multinomial output
            return -1*np.log(y_hat[np.arange(len(y_hat)), y.argmax(axis=1)]).sum()
        else: #use mean squared error
            return np.sum(np.square(np.linalg.norm(y - y_hat, axis=1))) / (2*len(x))
        

    def _numerical_grad(self, x, y, epsilon):
        '''_numerical_grad
        Computes grad_w, grad_b numerically. This
        is to be used for gradient checking primarily.
        The method uses the central difference:

            f'(x) = (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)

        >>> import nonlinearity as nl
        >>> nn = FeedForwardNet(layers=[3,2,3,1], transforms=[nl.ReLu, nl.ReLu, nl.Sigmoid], cost='mse')
        >>> nn.weights = [np.array([[-3,3],[2,5],[10,2]]), np.array([[-2,3,12],[6,1,2]]), np.array([[0.001],[1],[0.25]])]
        >>> nn.biases = [np.array([0,0]),np.array([-10,3,-5]),np.array([-20])]
        >>> for l in range(len(nn.weights)):
        ...     nn.weights[l] = nn.weights[l].astype('float64')
        ...     nn.biases[l] = nn.biases[l].astype('float64')
        >>> x = np.array([[1,2,-1]])
        >>> y_hat = nn._forward(x) # compute forward pass
        >>> grad_w, grad_b = nn._numerical_grad(x, 1, 1e-8)
        >>> print(grad_w)
	[array([[ 0.        , -0.16684528],
	       [ 0.        , -0.33369056],
	       [ 0.        ,  0.16684528]]), array([[ 0.        ,  0.        ,  0.        ],
	       [-0.00121866, -1.21865738, -0.30466435]]), array([[-6.20407394],
	       [-1.5510185 ],
	       [-1.88337959]])]
        >>> print(grad_b)
	[array([ 0.        , -0.16684528]), array([ -1.10775278e-04,  -1.10787041e-01,  -2.76967588e-02]), array([-0.11078704])]

	>>> import nonlinearity as nl
        >>> nn = FeedForwardNet(layers=[2,10,10,3,5], transforms=[nl.ReLu, nl.Tanh, nl.ReLu, nl.Softmax])
	>>> x = np.array([[100,10], [60,-30], [40, 20], [17, 10]]).astype('float64')
	>>> y = np.array([[1,0,0,0,0],[0,0,1,0,0],[0,1,0,0,0], [0,0,1,0,0]]).astype('float64')
	>>> y_hat = nn._forward(x)
	>>> grad_w, grad_b = nn._backward(x,y)
	>>> num_g_w, num_g_b = nn._numerical_grad(x, y, 1e-6)
	>>> for i in range(len(grad_w)):
	... 	print(np.linalg.norm(grad_w[i] - num_g_w[i]) / np.linalg.norm(grad_w[i] + num_g_w[i]) < 1e-6)
	... 	print(np.linalg.norm(grad_b[i] - num_g_b[i]) / np.linalg.norm(grad_b[i] + num_g_b[i]) < 1e-6)
        True
        True
        True
        True
        True
        True
        True
        True
        '''
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        for l in range(len(self.weights)):
            purturb = np.zeros_like(self.weights[l])
            for (i,j) in np.ndindex(self.weights[l].shape):
                purturb[i,j] = epsilon
                self.weights[l] += purturb
                right = self.cost(x,y)
                self.weights[l] -= 2*purturb
                left = self.cost(x,y)
                grad_w[l][i,j] = (right - left) / (2*epsilon)
                self.weights[l] += purturb
                purturb[i,j] = 0.0
                
        for l in range(len(self.biases)):
            purturb = np.zeros_like(self.biases[l])
            for i in np.ndindex(self.biases[l].shape):
                purturb[i] = epsilon
                self.biases[l] += purturb
                right = self.cost(x,y)
                self.biases[l] -= 2*purturb
                left = self.cost(x,y)
                grad_b[l][i] = (right - left) / (2*epsilon)
                self.biases[l] += purturb
                purturb[i] = 0.0
        return grad_w, grad_b
