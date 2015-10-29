__author__ = 'Giovanni'
# coding=utf-8
__author__ = 'Giovanni'
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class DropOutConvLayer(object):
    def __init__(self, input, srng, filter_shape, is_training,p):
        self.output = np.multiplyinput(input,srng.binomial(size=(filter_shape),p=p))


        def drop(input, srng, p= 0.4):
            """
            :type input: numpy.array
            :param input: layer or weight matrix on which dropout resp. dropconnect is applied

            :type p: float or double between 0. and 1.
            :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.

            """
            mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
            return input * mask








