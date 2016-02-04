__author__ = 'Giovanni'
import DataLoader
import theano.tensor as T
import theano
import LoadDataFunctions
import numpy as np

class BatchManager(object):
    def __init__(self, batchSize, superBatchSize,fileReferenceData, basePath,randomEveryEpoch=False):
        """
        :param batchSize: numero de registros por cada batch
        :param superBatchSize: numero de registros por cada super Batch
        :return:
        """
        self.noTotalBatchesForAllSuperBatches = -1
        self.noValidBatchesInSuperBatch = -1
        self.NoTotalSuperBatches = -1
        self.dataLoader = None
        self.batch = None
        self.batchSize = batchSize
        self.currentX = None
        #self.currentYFloats = None
        self.currentY = None


        #cada noValidBatchesInSuperBatch se tienen que cargar un nuevo dataSet
        self.noValidBatchesInSuperBatch = superBatchSize // batchSize
        if ((superBatchSize % batchSize) != 0):
            self.noValidBatchesInSuperBatch=self.noValidBatchesInSuperBatch + 1
        self.dataLoader = DataLoader.DataLoader(fileReferenceData, basePath,randomEveryEpoch)
        self.NoTotalSuperBatches = self.dataLoader.no_total_batches
        self.noTotalBatchesForAllSuperBatches=self.noValidBatchesInSuperBatch * self.dataLoader.no_total_batches

    def UpdateCurrentXAdYByBatchIndex(self, indexBatch):
        """
        :param indexBatch: es un numero que puede ser muy grande, ya que estos batches estan contenidos en los superbatches, y su numeracion atraviesa estos super batches, debe comenzar en 0 y no en 1
        :return:
        """

        if (indexBatch < self.noTotalBatchesForAllSuperBatches):
            superBatchIndexRequested = indexBatch // self.noValidBatchesInSuperBatch
            self.UpdateCurrentXAndY(superBatchIndexRequested)
        else:
            print "Index batch requested out of range"


    def UpdateCurrentXAndY(self, indexSuperBatch):
        if (self.currentX == None and self.currentY ==None):
            self.currentX,self.currentY=LoadDataFunctions.just_shared_dataset(self.dataLoader.getDataSetByBatchIndex(indexSuperBatch))
        else:
            rawX,rawY=self.dataLoader.getDataSetByBatchIndex(indexSuperBatch)
            if (self.dataLoader.IsNewDataSet == True):
                self.currentX.set_value(np.asarray(rawX,dtype=theano.config.floatX),borrow=True)
                self.currentY.set_value(np.asarray(rawY,dtype=theano.config.floatX),borrow=True)
                print ("raw DataSet Loaded in GPU Memory")

    def getTensorDataSet(self, superBatchIndexRequested):
        '''
        Primero retorna X y despues Y, retorna array's Tensor de theno
        '''
        rawData=self.dataLoader.getDataSetByBatchIndex(superBatchIndexRequested)
        return LoadDataFunctions.shared_dataset(rawData)

    def getRawDataSet(self, superBatchIndexRequested):
        return self.dataLoader.getDataSetByBatchIndex(superBatchIndexRequested)

    def change(self):
        self.currentX+=20
    def __unicode__(self,index):
        return unicode(self.getBatchByIndex(index))