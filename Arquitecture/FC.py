__author__ = 'Giovanni'


import numpy

import theano
import theano.tensor as T
import numpy as np

class FC(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out,title,initSD):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.title=title
        initMean  = 0
        numberWeights= numpy.prod((n_in,n_out))
        normalDistributionValues = numpy.random.normal(initMean, initSD, numberWeights)
        self.normalDistributionValues = normalDistributionValues.reshape((n_in,n_out))
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            numpy.asarray(
                self.normalDistributionValues,
                dtype=theano.config.floatX
            ),
            name='WFC',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.ones(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='bFC',
            borrow=True
        )

        self.bFast = theano.shared(
            value=numpy.ones(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='bFC',
            borrow=True
        )
        # Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes
        # Softmax regression allows us to handle }  where K  is the number of classes.
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        #el producto punto nos da un shape de (NoImages x 10575), con al funcion softmax, a cada peso(de 10575) por imagen,
        # se le asigna una probabilidad con respecto de de los otros pesos de la misma imagen, asi podremos elegir el peso mas alto en y_pred
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        ## T.argmax() is used to return the index of maximum value along a given axis.
        ## es nuestra funcion hipotesis
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1
        self.L2_sqr = (
                    (self.W ** 2).sum()
                )

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """
        checamos la probabilidad predecida para Y, despues sumamos todas las probabilidades y les sacamos el promedio de que tanto se equivoca
        Cost Function with mean and not sum

        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        return -T.mean(T.log(self.p_y_given_x) [T.arange(y.shape[0]), y])

        # end-snippet-2

    def negative_log_likelihoodTest(self,y):
        return -T.mean(T.log(self.p_y_given_x) [T.arange(y.shape[0]), y])
        #return T.log(self.p_y_given_x) # [T.arange(y.shape[0]), y]


    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        return T.mean(T.neq(self.y_pred, y))

    def seeY(self, y):
        return y

    def LoadWeights(self,weights):
        self.W.set_value(weights[0],borrow=True)
        self.b.set_value(weights[1],borrow=True)