__author__ = 'Giovanni'
import LoadDataFunctions

class GetData(object):
    def __init__(self, batch_size, no_total_batches, no_max_batches_loaded_in_range):
        self.batch_size= batch_size
        self.no_max_batches_loaded = no_max_batches_loaded_in_range

        self.currentRangeIndexLoded = -1
        self.no_total_batches = no_total_batches
        self.noRegistersByRange = no_max_batches_loaded_in_range * batch_size

    def getByBatchIndex(self, batchIndex_requested):
        if (batchIndex_requested <= self.no_total_batches):
            candidateRangeIndexLoded = (batchIndex_requested / self.no_max_batches_loaded)
            if (self.currentRangeIndexLoded != candidateRangeIndexLoded):
                #Se necesitan cargar nuevos datos
                self.currentRangeIndexLoded = candidateRangeIndexLoded
                dataSet= LoadDataFunctions.getDataSet(self.noRegistersByRange,self.currentRangeIndexLoded)

            offsetBatchIndex = batchIndex_requested - (self.currentRangeIndexLoded * self.no_max_batches_loaded)
            return dataSet[offsetBatchIndex * self.batch_size: (offsetBatchIndex + 1) * self.batch_size]
        else:
            #Error



