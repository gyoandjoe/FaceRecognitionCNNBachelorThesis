__author__ = 'Giovanni'
import DataLoader

class BatchManager(object):
    def __init__(self, batchSize, superBatchSize,fileReferenceData, basePath):
        """
        :param batchSize: numero de registros por cada batch
        :param superBatchSize: numero de registros por cada super Batch
        :return:
        """
        self.noValidBatchesInSuperBatch = -1
        self.NoTotalSuperBatches = -1
        self.dataLoader = None
        self.batchSize = batchSize
        if ((superBatchSize % batchSize) == 0):
            #cada noValidBatchesInSuperBatch se tienen que cargar un nuevo dataSet
            self.noValidBatchesInSuperBatch = superBatchSize / batchSize
            self.dataLoader = DataLoader.DataLoader(fileReferenceData, basePath)
            self.NoTotalSuperBatches = self.dataLoader.no_total_batches
            self.noTotalBatchesForAllSuperBatches=self.noValidBatchesInSuperBatch * self.dataLoader.no_total_batches
        else:
            print "Error"
    def getBatchByIndex(self, indexBatch):
        """
        :param indexBatch: es un numero que puede ser muy grande, ya que estos batches estan contenidos en los superbatches, y su numeracion atraviesa estos super batches, debe comenzar en 0 y no en 1

        :return:
        """
        batch = None
        if (indexBatch < self.noTotalBatchesForAllSuperBatches):
            superBatchIndexRequested = indexBatch // self.noValidBatchesInSuperBatch
            noBatchesUntilSuperBatchRequested = superBatchIndexRequested * self.noValidBatchesInSuperBatch
            offsetBatchIndex = indexBatch - noBatchesUntilSuperBatchRequested
            batch = self.dataLoader.getDataSetByBatchIndex(superBatchIndexRequested)[offsetBatchIndex * self.batchSize: (self.batchSize + 1) * self.batchSize]

        return batch



