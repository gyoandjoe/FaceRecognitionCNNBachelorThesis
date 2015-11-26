__author__ = 'Giovanni'

#import ConvReluLayer
from theano.tensor.signal import downsample
from theano import tensor
import theano
import numpy as np
import DualConvReluLayer
from Arquitecture import DualConvReluAndConvLayer
import DropOutConvLayer
import FC


class FRCNN(object):
    def __init__(self, rng_droput, is_training, img_input, noImages, pdrop=0.4):

        self.img_input= img_input
        self.Conv_11_12 = DualConvReluLayer.DualConvReluLayer(
            input=img_input,
            filter_shape=(32, 1, 3, 3),
            image_shape=(noImages, 1, 100, 100)
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
            image_shape=(noImages, 64, 50, 50)
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
            image_shape=(noImages, 128, 25, 25)
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
            image_shape=(noImages, 192, 13, 13)
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
            image_shape=(noImages, 256, 7, 7)
        )

        self.pooled_5 = downsample.max_pool_2d(
            input=self.Conv_51_52.output,
            st=(1, 1), #stride
            ds=(7, 7),
            mode='average_inc_pad',
            ignore_border=False
        )

        self.dropout = DropOutConvLayer.DropOutConvLayer(
            input= tensor.reshape(self.pooled_5,(noImages,320)),
            srng=rng_droput,
            image_shape=(noImages, 320),
            is_training=is_training,
            p=pdrop
        )

        self.FC = FC.FC(
            input=self.dropout.output,
            n_in=320,
            n_out=10575
        )
