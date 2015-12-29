'''nonlinearity
file creates classes for common neural network
nonlinearities. All classes have 2 static methods:

    f(x numpy.ndarray) -> numpy.ndarray
        evaluate the nonlinearity on a numpy array

    df(y numpy.ndarray) -> numpy.ndarray
        evaluate the nonlinearity on a numpy array
        _FROM THE OUTPUT_ of f(x)
'''
import numpy as np
from scipy.misc import logsumexp
from math import tanh, exp

class Tanh():
    '''Tanh
    hyperbolic tangent activation function

    f(x) = tanh(x)
    f'(x) = 1 - tanh^2(x)
    '''

    @staticmethod
    def f(x):
        '''f
        Tanh function evaluated
        at x

        >>> import numpy
        >>> Tanh.f(np.array([-999, 0, 999]))
        array([-1.,  0.,  1.])
        '''
        return np.tanh(x)

    @staticmethod
    def df(x):
        '''df
        returns the first derivative
        of the tanh function at x

        d/dx tanh(x) = 1 - tanh^2(x)

        >>> import numpy
        >>> Tanh.df(Tanh.f(np.array([-999, 0, 999])))
        array([ 0.,  1.,  0.])
        '''
        return 1 - np.square(x)

class Sigmoid():
    '''Sigmoid
    sigmoid activatoin function

    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) (1 - f(x))
    '''

    @staticmethod
    def f(x):
        '''f
        sigmoid function evaluated at x

        >>> import numpy
        >>> Sigmoid.f(numpy.array([-999, 0, 999]))
        array([ 0. ,  0.5,  1. ])
        '''
        return np.reciprocal(1 + np.exp(-1*x))

    @staticmethod
    def df(x):
        '''df
        sigmoid function first derivatve
        evaluated at x

        >>> import numpy
        >>> _ = numpy.seterr(over='ignore', under='ignore')
        >>> Sigmoid.df(Sigmoid.f(np.array([-999, 0, 999])))
        array([ 0.  ,  0.25,  0.  ])
        '''
        return x * (1 - x)

class ReLu():
    '''ReLu
    rectified linear unit activation

    f(x) = max( 0, x )
    f'(x) = 1 if x > 0; 0 otherwise
    '''

    @staticmethod
    def f(x):
        '''f
        ReLu activation function evaluated
        at x

        >>> import numpy
        >>> ReLu.f(numpy.array([-10, -1, 1, 10]))
        array([ 0,  0,  1, 10])
        '''
        return np.maximum(x, 0)

    @staticmethod
    def df(x):
        '''df
        ReLu derivative of f evaluated
        at x

        >>> import numpy
        >>> ReLu.df(ReLu.f(numpy.array([-10, -1, 1, 10]))) + 0 # add 0 to case from bool to int
        array([0, 0, 1, 1])
        '''
        return x > 0

class Identity():
    '''Identity
    identity activation function

    f(x) = x
    f'(x) = 1
    '''

    @staticmethod
    def f(x):
        '''f
        Identity activation function evaluated
        at x

        >>> import numpy
        >>> Identity.f(numpy.array([-1,0,1]))
        array([-1,  0,  1])
        '''
        return x

    @staticmethod
    def df(x):
        '''df
        Identity activation first derivative
        evaluated at x

        >>> import numpy
        >>> Identity.df(Identity.f(numpy.array([-1,0,1])))
        array([1, 1, 1])
        '''
        return np.ones_like(x)

class Softmax():
    '''Softmax
    softmax output function

    uses a numerically stable variant to prevent
    float overflow (not underflow) from
        http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    f(x)_i = exp(x_i) / sum_j( exp(x) )
    '''

    @staticmethod
    def f(x):
        '''f
        Softmax output function evaluated
        with a numpy array x

        >>> Softmax.f(np.array([-999, 1, 1]))
        array([ 0. ,  0.5,  0.5])

        >>> Softmax.f(np.array([[-999,1,1],[1,-999,1],[1,1,-999],[1,1,1]]))
	array([[ 0.        ,  0.5       ,  0.5       ],
	       [ 0.5       ,  0.        ,  0.5       ],
	       [ 0.5       ,  0.5       ,  0.        ],
	       [ 0.33333333,  0.33333333,  0.33333333]])
        '''
        x = x.astype('float64')
        shape = x.shape
        if len(x.shape) == 1:
            x.shape = (1,x.shape[0])
        mx =  np.maximum(0.0, x.max(axis=1))
        mx.shape = (1, mx.shape[0])
        x -= mx.T
        x -= logsumexp(x, axis=1, keepdims=True)
        x.shape = shape
        return np.exp(x)

    @staticmethod
    def df(x):
        '''df
        Softmax derivative with respect to x

        (used maximum entropy so derivative is 1)

        >>> Softmax.df(Softmax.f(np.array([-100, 0, 1])))
        array([ 1.,  1.,  1.])
        '''
        return np.ones_like(x)

class LogSoftmax():
    '''LogSoftmax
    logarithm of softmax output function. This
    output can be used in the same way as softmax,
    with the same decision parameters, but instead
    of returning probabilities you return log-probabilities.
    To convert back to probabilities just exponentiate
    the output.

    uses a numerically stable variant to prevent
    float overflow (not underflow)

    f(x)_i = log( exp(x_i) / sum_j( exp(x) ) )
    '''

    @staticmethod
    def f(x):
        '''f
        LogSoftmax output function evaluated
        with a numpy array x

        >>> LogSoftmax.f(np.array([-999, 1, 1]))
        array([ -1.00069315e+03,  -6.93147181e-01,  -6.93147181e-01])

        >>> LogSoftmax.f(np.array([[-999,1,1],[1,-999,1],[1,1,-999]]))
        array([[ -1.00069315e+03,  -6.93147181e-01,  -6.93147181e-01],
               [ -6.93147181e-01,  -1.00069315e+03,  -6.93147181e-01],
               [ -6.93147181e-01,  -6.93147181e-01,  -1.00069315e+03]])
        '''
        x = x.astype('float64')
        shape = x.shape
        if len(x.shape) == 1:
            x.shape = (1,x.shape[0])
        mx =  np.maximum(0.0, x.max(axis=1))
        mx.shape = (1, mx.shape[0])
        x -= mx
        res = x - logsumexp(x, axis=1)
        res.shape = shape
        return res

    @staticmethod
    def df(x):
        '''df
        LogSoftmax derivative with respect to x

        (used maximum entropy so derivative is 1)

        >>> LogSoftmax.df(LogSoftmax.f(np.array([-100, 0, 1])))
        array([ 1.,  1.,  1.])
        '''
        return np.ones_like(x)
