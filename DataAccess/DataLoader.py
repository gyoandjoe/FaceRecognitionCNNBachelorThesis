import cPickle
import random

__author__ = 'Giovanni'
import numpy as np
import os


class DataLoader(object):
    def __init__(self, fileReferenceData, basePath,withshufledRows = False):
        self.currentBatchIndexLoded = -1
        self.basePath= basePath
        self.fileReferenceData = fileReferenceData
        self.withshufledRows = withshufledRows
        self.currentDataSet = None
        self.IsNewDataSet = True
        self.WithshufledRows = withshufledRows
        self.dataSetBatchList = np.loadtxt( self.fileReferenceData,delimiter=',', dtype='str',usecols = (0,1))#Matriz con dos columnas, el index, y el nombre del archivo
        self.no_total_batches =  self.dataSetBatchList.shape[0]  #Calculado del archivo
        self.NoBatchesLoaded = 0
        self.BatchIndexes = range(self.no_total_batches)
        self.getFilesOfReference()


    def getFilesOfReference(self):

        if self.withshufledRows == True:
            random.shuffle(self.BatchIndexes)
            print ("Shuffled DataSet")

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
        if (self.NoBatchesLoaded == self.no_total_batches):
            self.NoBatchesLoaded = 0
            self.getFilesOfReference()

        fLoaded = file(os.path.join(self.basePath,self.dataSetBatchList[batchIndex][1]), 'rb')

        data = cPickle.load(fLoaded)
        fLoaded.close()

        #if self.WithshufledRows == True:
            #Debemos crear un array de 0 hasta n examples in data, despues desordenarlo y usarlo como referencia para reordenar daaX y dataY
        dataX = data[0] #np.asarray(data[0])
        dataY = data[1] #np.asarray(data[1])
        self.NoBatchesLoaded = self.NoBatchesLoaded + 1
        print "raw dataSet" + self.dataSetBatchList[self.BatchIndexes[batchIndex]][1] + " --- LOADED!"
        return dataX, dataY






