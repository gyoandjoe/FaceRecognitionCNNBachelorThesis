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
    def __init__(self, rng_droput, is_training, img_input, noImages, logManager, pdrop=0.4):

        self.LogManager = logManager
        self.img_input= img_input

        self.Conv_11_12 = DualConvReluLayer.DualConvReluLayer(
            input=img_input,
            filter_shape=(32, 1, 3, 3),
            image_shape=(noImages, 1, 100, 100),
            titleLayer0='Conv_11',
            titleLayer1='Conv_12'
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
            image_shape=(noImages, 64, 50, 50),
            titleLayer0='Conv_21',
            titleLayer1='Conv_22'
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
            image_shape=(noImages, 128, 25, 25),
            titleLayer0='Conv_31',
            titleLayer1='Conv_32'
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
            image_shape=(noImages, 192, 13, 13),
            titleLayer0='Conv_41',
            titleLayer1='Conv_42'
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
            image_shape=(noImages, 256, 7, 7),
            titleLayer0='Conv_51',
            titleLayer1='Conv_52'
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
            n_out=10575,
            title="FC"
        )

    def saveState(self,logId, values):
        self.saveTrainValues(logId, values)
        self.saveCNNWeights(logId)

    def saveTrainValues(self, logId, values):
        self.LogManager.saveState_TrainValues(logId,values)


    def saveCNNWeights(self,logId):
        self.saveStandarCNNLayer(logId,self.Conv_11_12.layer0)
        self.saveStandarCNNLayer(logId,self.Conv_11_12.layer1)

        self.saveStandarCNNLayer(logId,self.Conv_21_22.layer0)
        self.saveStandarCNNLayer(logId,self.Conv_21_22.layer1)

        self.saveStandarCNNLayer(logId,self.Conv_31_32.layer0)
        self.saveStandarCNNLayer(logId,self.Conv_31_32.layer1)

        self.saveStandarCNNLayer(logId,self.Conv_41_42.layer0)
        self.saveStandarCNNLayer(logId,self.Conv_41_42.layer1)

        self.saveStandarCNNLayer(logId,self.Conv_51_52.layer0)
        self.saveStandarCNNLayer(logId,self.Conv_51_52.layer1)

        self.saveStandarCNNLayer(logId,self.FC)

        print "DataSet Weighs Saved"

    def GetAndLoadState(self, logId):
        self.LoadWeightFromLogId(logId)
        return self.LogManager.loadState_TrainValues(logId)

    def LoadWeightFromLogId(self, logId,fileWeights="None", basePath="None"):
        self.loadCNNWeights(logId, fileWeights, basePath)

    def loadCNNWeights(self,logId,fileWeights="None", basePath="None"):
        self.LoadStandarCNNLayer(logId,self.Conv_11_12.layer0, fileWeights, basePath)
        self.LoadStandarCNNLayer(logId,self.Conv_11_12.layer1, fileWeights, basePath)

        self.LoadStandarCNNLayer(logId,self.Conv_21_22.layer0, fileWeights, basePath)
        self.LoadStandarCNNLayer(logId,self.Conv_21_22.layer1, fileWeights, basePath)

        self.LoadStandarCNNLayer(logId,self.Conv_31_32.layer0, fileWeights, basePath)
        self.LoadStandarCNNLayer(logId,self.Conv_31_32.layer1, fileWeights, basePath)

        self.LoadStandarCNNLayer(logId,self.Conv_41_42.layer0, fileWeights, basePath)
        self.LoadStandarCNNLayer(logId,self.Conv_41_42.layer1, fileWeights, basePath)

        self.LoadStandarCNNLayer(logId,self.Conv_51_52.layer0, fileWeights, basePath)
        self.LoadStandarCNNLayer(logId,self.Conv_51_52.layer1, fileWeights, basePath)

        self.LoadStandarCNNLayer(logId,self.FC, fileWeights, basePath)

        print "DataSet Weighs Loaded"

    def LoadStandarCNNLayer(self,logId,convStandarCNNLayer,fileWeights="None", basePath="None"):
        rawData = self.LogManager.loadState_CNNLayerWeights(convStandarCNNLayer.title,logId,fileWeights, basePath)
        convStandarCNNLayer.LoadWeights( (np.asarray(rawData[0],dtype=theano.config.floatX),np.asarray(rawData[1],dtype=theano.config.floatX)))

    def saveStandarCNNLayer(self, logid, convStandarCNNLayer):
        self.LogManager.saveState_CNNLayerWeights(
            data=(np.asarray(convStandarCNNLayer.params[0].get_value()), np.asarray(convStandarCNNLayer.params[1].get_value())),
            logId=logid,
            title=convStandarCNNLayer.title
        )