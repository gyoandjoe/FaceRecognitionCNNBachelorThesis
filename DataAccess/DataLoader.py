import cPickle

__author__ = 'Giovanni'
import numpy as np
import os
import LoadDataFunctions

class DataLoader(object):
    def __init__(self, fileReferenceData, basePath):
        self.currentBatchIndexLoded = -1
        self.basePath= basePath
        self.dataSetBatchList = np.loadtxt( fileReferenceData,delimiter=',', dtype='str',usecols = (0,1))#Matriz con dos columnas, el index, y el nombre del archivo
        self.no_total_batches =  self.dataSetBatchList.shape[0]  #Calculado del archivo
        self.currentDataSet = None

    def getDataSetByBatchIndex(self, batchIndex_requested):
        if (batchIndex_requested < self.no_total_batches):
            if (self.currentBatchIndexLoded != batchIndex_requested):
                #Se necesitan cargar nuevos datos
                self.currentDataSet=self.getTensorDataSet(batchIndex_requested)
                self.currentBatchIndexLoded = batchIndex_requested
            return self.currentDataSet
        else:
            print "ERROR -- El index solicitado esta fuera de rango :( "


    def getDataSet(self,batchIndex):
        '''
        Primero returna X y despues Y, retorna arrays de numpy
        '''
        fLoaded = file(os.path.join(self.basePath,self.dataSetBatchList[batchIndex][1]), 'rb')
        data = cPickle.load(fLoaded)
        fLoaded.close()
        dataX = np.asarray(data[0])
        dataY = np.asarray(data[1])

        return dataX, dataY

    def getTensorDataSet(self, batchIndex):
        '''
        Primero returna X y despues Y, retorna arrays Tensor de theno
        '''
        return LoadDataFunctions.shared_dataset(self.getDataSet(batchIndex))




