import cPickle

__author__ = 'Giovanni'
import numpy as np
import os


class DataLoader(object):
    def __init__(self, fileReferenceData, basePath):
        self.currentBatchIndexLoded = -1
        self.basePath= basePath
        self.dataSetBatchList = np.loadtxt( fileReferenceData,delimiter=',', dtype='str',usecols = (0,1))#Matriz con dos columnas, el index, y el nombre del archivo
        self.no_total_batches =  self.dataSetBatchList.shape[0]  #Calculado del archivo
        self.currentDataSet = None
        self.IsNewDataSet = True

    def getDataSetByBatchIndex(self, batchIndex_requested):
        self.IsNewDataSet = False
        if (batchIndex_requested < self.no_total_batches):
            if (self.currentBatchIndexLoded != batchIndex_requested):
                #Se necesitan cargar nuevos datos
                self.IsNewDataSet = True
                self.currentDataSet= self.getDataSet(batchIndex_requested) #No SHARED
                #currentDataSet=self.getTensorDataSet(batchIndex_requested) #SHARED
                self.currentBatchIndexLoded = batchIndex_requested

            return self.currentDataSet
        else:
            print "ERROR -- El index solicitado esta fuera de rango :( "


    def getDataSet(self,batchIndex):
        '''
        Primero retorna X y despues Y, retorna arrays de numpy
        '''
        fLoaded = file(os.path.join(self.basePath,self.dataSetBatchList[batchIndex][1]), 'rb')
        data = cPickle.load(fLoaded)
        fLoaded.close()
        dataX = data[0] #np.asarray(data[0])
        dataY = data[1] #np.asarray(data[1])

        return dataX, dataY






