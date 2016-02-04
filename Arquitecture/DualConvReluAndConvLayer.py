# coding=utf-8
__author__ = 'Giovanni'
from Arquitecture import ConvLayer, ConvReluLayer


class DualConvReluAndConvLayer(object):
    def __init__(self, input, filter_shape, image_shape,titleLayer0,titleLayer1):
        assert image_shape[1] == filter_shape[1]
        no_filters_l0 = filter_shape[0]

        #No input feature maps  o deep de cada imagen
        D0 = filter_shape[1]
        height_filter = filter_shape[2]
        weight_filter = filter_shape[3]

        noimages = image_shape[0]
        heightimage = image_shape[2]
        weightimage = image_shape[3]

        self.layer0 = ConvReluLayer.ConvReluLayer(
            input=input,
            filter_shape=(no_filters_l0, D0, weight_filter, height_filter),
            image_shape=(noimages, D0, heightimage, weightimage),
            title=titleLayer0,
            initSD=0.001
        )

        nfilters_l1 = no_filters_l0 * 2

        #No input feature maps  o deep de cada imagen
        d1 = no_filters_l0
        self.layer1 = ConvLayer.ConvLayer(
            input=self.layer0.output,
            filter_shape=(nfilters_l1, d1, weight_filter, height_filter),
            image_shape=(noimages, d1, heightimage, weightimage),
            title=titleLayer1,
            initSD=0.001

        )

        self.output = self.layer1.output