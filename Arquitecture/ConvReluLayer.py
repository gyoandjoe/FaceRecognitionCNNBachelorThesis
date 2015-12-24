# coding=utf-8
__author__ = 'Giovanni'
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class ConvReluLayer(object):
    def __init__(self, input, filter_shape, image_shape,title):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: Imagen(volumen) de entrada para este volumen convolucional, el shape esta especificado en image_shape

        :type image_shape: lista de longitud de tamaño 4
        batch size: numero de imagenes y por cada imagen existe un numero de feature maps
        num input feature maps: Numero de feature maps de entrada, es la profundidad del volumen de entrada
        ambos batch size y num input feature maps, indican el numero de imagenes
        :param image_shape: (batch size, num input feature maps, image height, image width)

        :type filter_shape: lista de longitud 4
        -Numero de filtros o kernels, ambos kernels deben ser de la misma profundidad
        -Numero de feature maps de entrada: es la profundidad del volumen de entrada, este numero debe coincidir con la profundidad(no especificada aqui) de los kernels
        numero de filtros o kernels y num input feature maps indican el numero de filtros, pero el num de feature maps de entrada esta asociado al deep de cada imagen de entrada
        :param filter_shape: (numero de filtros o kernels, num input feature maps, filter height, filter width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        #El numero de feature maps deben coincidir tanto los especificados en image shape como en filter shape
        assert image_shape[1] == filter_shape[1]

        self.title = title
        self.input = input

        # There are "num input feature maps(profundidad de volumen de entrada) * filter height * filter width"
        # numero de entradas(del volumen de entrada)  para producir cada unidad en el volumen de salida
        numberUnits_in = np.prod(filter_shape[1:])

        # each unit in the lower(next) layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /  pooling size
        # (Numero de kernels(No Filters) * tamaño del volumen de salida) / tamaño de subsampling por pixel
        #numberUnits_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))


        # mean and standard deviation
        initMean, initSD = 0, 0.01
        numberWeights= np.prod(filter_shape[0:])
        self.normalDistributionValues = np.random.normal(initMean, initSD, numberWeights)
        print "NumberOfWeight in ConvReluLayer("+self.title+"): "+ str(numberWeights)
        self.normalDistributionValues = self.normalDistributionValues.reshape(filter_shape)

        #debug
        #np.savetxt('filter.csv', self.normalDistributionValues[1][0],'%5.20f' ,delimiter=',')

        # initialize weights with normal distribution (gausian distribution)
        #W_bound = np.sqrt(6. / 100)
        self.W = theano.shared(
            np.asarray(
                self.normalDistributionValues,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # convolve input feature maps with filters
        #return Tensor of filtered images, with shape ([number images,] [number filters,] image height, image width).
        """
        Formula para tamaño de salida de la convolucion(debe ser un entero de lo contrario no sera valida):
        W: ancho o alto de volumen de entrada
        F: tamaño del campo receptivo local(lado de kernel)
        P: padding que se aplica al volumen de entrada, se rellena con 0 si es que hay padding
        S: Stride, que se aplica al momento de hacer la convolucion, numero de pixeles que se salta de conv en conv
        ((W−F+2P)/S) + 1 = entero
        """
        conv_out = conv.conv2d(
            input=input,
            subsample=(1,1),#Stride
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            border_mode='full'
        ) [:,:,1:image_shape[2]+1,1:image_shape[2]+1]
        self.outputJustConv = conv_out

        #Se aplica BIAS a cada elemento de la convolucion resultante, un BIAS diferente  por kernel, y con self.b.dimshuffle lo colocamos en posicion correcta para hacer la suma con conv_out
        b_values = np.ones((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        #Aplicamos RELU a (conv_out + BIAS)
        self.output = self.Relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def Relu(self,x):
        return theano.tensor.switch(x < 0, 0, x)


    def LoadWeights(self,weights):

        self.W.set_value(weights[0],borrow=True)
        self.b.set_value(weights[1],borrow=True)



