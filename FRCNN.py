__author__ = 'Giovanni'
import theano
import theano.tensor as T
from PIL import Image
import numpy as np

#import ConvReluLayer
import DualConvReluLayer
import DualConvReluAndConvLayer
from theano.tensor.signal import downsample

class FRCNN(object):
    def __init__(self, rng_droput, is_training, img_input, pdrop=0.4):

        self.Conv_11_12 = DualConvReluLayer.DualConvReluLayer(
            input=img_input,
            filter_shape=(32, 1, 3, 3),
            image_shape=(1, 1, 100, 100)
        )

        # downsample each feature map individually, using maxpooling
        self.pooled_1 = downsample.max_pool_2d(
            input=self.Conv_11_12.output,
            st=(2, 2), #stride
            ds=(2, 2),
            mode='max',
            ignore_border=False
        )

        self.Conv_21_22 = DualConvReluLayer.DualConvReluLayer(
            input=self.pooled_1,
            filter_shape=(64, 64, 3, 3),
            image_shape=(1, 64, 50, 50)
        )

        self.pooled_2 = downsample.max_pool_2d(
            input=self.Conv_21_22.output,
            st=(2, 2), #stride
            ds=(2, 2),
            mode='max',
            ignore_border=False
        )

        self.Conv_31_32 = DualConvReluLayer.DualConvReluLayer(
            input=self.pooled_2,
            filter_shape=(96, 128, 3, 3),
            image_shape=(1, 128, 25, 25)
        )

        self.pooled_3 = downsample.max_pool_2d(
            input=self.Conv_31_32.output,
            st=(2, 2), #stride
            ds=(2, 2),
            mode='max',
            ignore_border=False
        )

        self.Conv_41_42 = DualConvReluLayer.DualConvReluLayer(
            input=self.pooled_3,
            filter_shape=(128, 192, 3, 3),
            image_shape=(1, 192, 13, 13)
        )

        self.pooled_4 = downsample.max_pool_2d(
            input=self.Conv_41_42.output,
            st=(2, 2), #stride
            ds=(2, 2),
            mode='max',
            ignore_border=False
        )

        self.Conv_51_52 = DualConvReluAndConvLayer.DualConvReluAndConvLayer(
            input=self.pooled_4,
            filter_shape=(160, 256, 3, 3),
            image_shape=(1, 256, 7, 7)
        )

        self.pooled_5 = downsample.max_pool_2d(
            input=self.Conv_51_52.output,
            st=(1, 1), #stride
            ds=(7, 7),
            mode='average_inc_pad',
            ignore_border=False
        )


is_training = T.iscalar('is_training')
x = T.matrix('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels




# dimensions are (height, width, channel)
img_input = x.reshape((1, 1, 100, 100))


#srng_droput =T.shared_randomstreams.RandomStreams(seed=12345)
random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

classifier = FRCNN(
    rng_droput=rng_droput,
    is_training=is_training,
    img_input=img_input
)

img = Image.open(r'E:\dev\TesisFRwithCNN\TestRS\0000117\005-l.jpg')
img = np.asarray(img, dtype=theano.config.floatX) / 256

train_model = theano.function(
    [],
      classifier.pooled_5,
      givens={
          x: img,
          is_training: np.cast['int32'](1)
      },
      on_unused_input='warn'
)

valTest = train_model()
valTest2 = train_model()
print ("...")