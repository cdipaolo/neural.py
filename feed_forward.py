import nonlinearity as nl

try:
    import cPickle as pickle
except:
    import pickle
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
        self.log_softmax_output = False
        if transforms[-1] == nl.LogSoftmax:
            self.log_softmax_output = True

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

        # list to save costs to
        self.costs = []

    def fit(self, x, y, lr=0.1, max_iter=1000, mini_batch_size=1, \
            save_accuracy=True, save_costs=False, save_model=False, \
            model_filename='/feed_forward_nn_{iteration}_iters_{accuracy}_accuracy.pkl', \
            report_every=10, test_x=None, test_y=None):
        '''fit
        fits the neural network to the dataset using
        minibatch gradient descent.

        x :: np.ndarray (1 or 2 dimensions)
        y :: np.ndarray (1 or 2 dimensions)
           - must have the same number of rows as x

        test_x :: np.ndarray (1 or 2 dimensions)
                - testing dataset which is only used
                  for accuracy checkin (not cost)
        test_y :: np.ndarray (1 or 2 dimensions)
                - must have the same number of rows as x

        lr :: float
            - learning rate alpha
        max_iter :: int
            - maximum number of iterations/epochs to
              run for
        save_costs :: bool
            - whether to record cost values with every
              update to self.costs
        save_accuracy :: bool
            - whether to check the predicted accuracy on the entire
              training set at each report_every iterations
        save_model :: bool
            - whether to persist the model to disk at each
              report_every itertions
        model_filename :: string
            - formattable string of the full filepath to save
              the model to. Possible keys to use in formatting
              include:
                iteration :: int (current iteration)
                accuracy :: int (floored percent, ie. 69 for 69.235% of accuracy)
                cost :: int (floored cost metric)
                lr :: float (learning rate lr)
        report_every :: int
            - number of iterations between each printed
              status update / cost save / model persist /
              accuracy check
        '''
        if len(x.shape) == 1:
            x.shape = (1,x.shape[0])
        if save_costs:
            self.costs = []
        m = len(x)
        minibatches = m // mini_batch_size
        n_weights = len(self.weights)
        for iteration in range(max_iter):
            # go through minibatches each epoch
            for i in range(minibatches):
                # get minibatch
                idx = np.random.choice(m, mini_batch_size)
                batch_x, batch_y = x[idx], y[idx]
                self._forward(batch_x)
                grad_w, grad_b = self._backward(batch_x, batch_y)

                # gradient descent step
                for l in range(n_weights):
                    self.weights[l] -= lr * grad_w[l]
                    self.biases[l] -= lr * grad_b[l]

            if iteration % report_every == 0:
                accuracy = 'Not Saved'
                cost = 'Not Saved'
                if save_costs:
                    self.costs.append(self.cost(x,y))
                    cost = self.costs[-1]
                if save_accuracy and test_x:
                    accuracy = self.evaluate(test_x, test_y)
                elif save_accuracy:
                    accuracy = self.evaluate(x, y)
                if save_model:
                    self.persist(model_filename.format(iteration=iteration+1, \
                            accuracy=accuracy, \
                            cost=cost, \
                            lr=lr))
                # print update
                if save_accuracy:
                    print('Epoch {}: {}% accuracy'.format(iteration+1, accuracy))
                elif save_costs:
                    print('Epoch {}: {} cost'.format(iteration+1, cost))
                else:
                    print('Epoch {} completed'.format(iteration+1))
        print('Training completed')

    def evaluate(self, x, y):
        '''evaluate
        evaluates a dataset and tags in a classification
        setting, returning the accuracy of the network

	>>> import nonlinearity as nl
	>>> np.random.seed(1)
        >>> nn = FeedForwardNet(layers=[2,10,10,3,5], transforms=[nl.ReLu, nl.Tanh, nl.ReLu, nl.Softmax])
	>>> x = np.array([[100,10], [60,-30], [40, 20], [17, 10]]).astype('float64')
	>>> y = np.array([[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0], [0,1,0,0,0]]).astype('float64')
        >>> nn._forward(x)
	array([[ 0.18337369,  0.30337132,  0.14642168,  0.11010731,  0.256726  ],
	       [ 0.06212841,  0.22033675,  0.31645162,  0.06891918,  0.33216404],
	       [ 0.1865579 ,  0.30425151,  0.14506023,  0.11126231,  0.25286805],
	       [ 0.19140122,  0.30549936,  0.14299832,  0.11299255,  0.24710855]])
	>>> nn.evaluate(x,y)
	0.75
        '''
        if len(x.shape) == 1:
            x.shape = (1, x.shape[0])
            y.shape = (1, y.shape[0])
        y_hat = self._forward(x)
        pred = (y_hat == np.max(y_hat, axis=1, keepdims=True))
        return pred[np.arange(len(pred)), y.argmax(axis=1)].sum() / len(x)

    def persist(self, filename):
        '''persist
        saves the model to a filepath specified
        using the Pickle format

	>>> import nonlinearity as nl
	>>> np.random.seed(1)
        >>> nn = FeedForwardNet(layers=[2,10,10,3,5], transforms=[nl.ReLu, nl.Tanh, nl.ReLu, nl.Softmax])
	>>> x = np.array([[100,10], [60,-30], [40, 20], [17, 10]]).astype('float64')
	>>> y = np.array([[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0], [0,1,0,0,0]]).astype('float64')
	>>> nn.evaluate(x,y)
	0.75
        >>> nn.persist('/tmp/model_test.pkl')
        >>> nn = None
        >>> with open('/tmp/model_test.pkl', 'rb') as f:
        ...     nn = pickle.load(f)
	>>> nn.evaluate(x,y)
        0.75
        '''
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)
        

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
        if self.x_entropy and not self.log_softmax_output: # assume using sigmoid or softmax
            self.deltas[-1] = (self.activations[-1] - y)
        elif self.x_entropy: # convert to softmax from log-softmax for learning
            self.deltas[-1] = (np.exp(self.activations[-1]) - y)
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
            if self.log_softmax_output: # don't log if already logged
                return -1*y_hat[np.arange(len(y_hat)), y.argmax(axis=1)].sum()
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
	>>> np.random.seed(42)
        >>> nn = FeedForwardNet(layers=[2,10,10,3,5], transforms=[nl.ReLu, nl.Tanh, nl.ReLu, nl.Softmax])
	>>> x = np.array([[100,10], [60,-30], [40, 20], [17, 10]]).astype('float64')
	>>> y = np.array([[1,0,0,0,0],[0,0,1,0,0],[0,1,0,0,0], [0,0,1,0,0]]).astype('float64')
	>>> y_hat = nn._forward(x)
	>>> grad_w, grad_b = nn._backward(x,y)
	>>> num_g_w, num_g_b = nn._numerical_grad(x, y, 1e-4)
	>>> for i in range(len(grad_w)):
	... 	print(np.linalg.norm(grad_w[i] - num_g_w[i]) / np.linalg.norm(grad_w[i] + num_g_w[i]) < 1e-4)
	... 	print(np.linalg.norm(grad_b[i] - num_g_b[i]) / np.linalg.norm(grad_b[i] + num_g_b[i]) < 1e-4)
        True
        True
        True
        True
        True
        True
        True
        True

	>>> import nonlinearity as nl
	>>> np.random.seed(42)
        >>> nn = FeedForwardNet(layers=[2,10,10,3,1], transforms=[nl.ReLu, nl.Tanh, nl.ReLu, nl.Sigmoid])
	>>> x = np.array([[100,10], [60,-30], [40, 20], [17, 10]]).astype('float64')
	>>> y = np.array([[1],[1],[0], [0]]).astype('float64')
	>>> y_hat = nn._forward(x)
	>>> grad_w, grad_b = nn._backward(x,y)
	>>> num_g_w, num_g_b = nn._numerical_grad(x, y, 1e-4)
	>>> for i in range(len(grad_w)):
	... 	print(np.linalg.norm(grad_w[i] - num_g_w[i]) / np.linalg.norm(grad_w[i] + num_g_w[i]) < 1e-4)
	... 	print(np.linalg.norm(grad_b[i] - num_g_b[i]) / np.linalg.norm(grad_b[i] + num_g_b[i]) < 1e-4)
	True
	True
	True
	True
	True
	True
	True
	True

	>>> import nonlinearity as nl
	>>> np.random.seed(42)
        >>> nn = FeedForwardNet(layers=[2,5,10,25,10], transforms=[nl.Sigmoid, nl.ReLu, nl.Tanh, nl.LogSoftmax])
	>>> x = np.array([[100,10], [60,-30], [40, 20], [-170, 10]]).astype('float64')
	>>> y = np.array([[1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0]]).astype('float64')
	>>> y_hat = nn._forward(x)
	>>> grad_w, grad_b = nn._backward(x,y)
	>>> num_g_w, num_g_b = nn._numerical_grad(x, y, 1e-4)
	>>> for i in range(len(grad_w)):
	... 	print(np.linalg.norm(grad_w[i] - num_g_w[i]) / np.linalg.norm(grad_w[i] + num_g_w[i]) < 1e-4)
	... 	print(np.linalg.norm(grad_b[i] - num_g_b[i]) / np.linalg.norm(grad_b[i] + num_g_b[i]) < 1e-4)
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
