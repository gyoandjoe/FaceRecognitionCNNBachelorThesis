__author__ = 'Giovanni'
import csv
import cPickle
import os

class LogManager(object):
    def __init__(self, basePath, id_fileWeights, id_fileLossValues, id_file_csvPerformance, arquitecture):
        self.basePath = basePath
        self.id_file_csvPerformance = id_file_csvPerformance
        self.id_fileWeights = id_fileWeights
        self.id_fileLossValues = id_fileLossValues

    def saveState_CNNLayerWeights(self, logId, data,title):
        f = file(self.basePath + "\\Weights_" + self.id_fileWeights + "_" + title + "_" + logId + '.pkl', 'w+b')
        cPickle.dump(data, f, protocol=2)
        f.close()
        return 0

    def loadState_CNNLayerWeights(self,title,logId):
        fLoaded = file(self.basePath + "\\Weights_" + self.id_fileWeights + "_" + title + "_" + logId + '.pkl', 'rb')
        data = cPickle.load(fLoaded)
        fLoaded.close()
        return data[0], data[1]

    def saveState_TrainProcessData(self, epoch, minibatch_index, best_validation_loss, best_iter, done_looping, patience):
        return 0

    def savePerformanceInfo(self, LogId, cost,validation_loss,test_score,epoch, minibatch_index, iter, best_validation_loss, best_iter, done_looping, patience):
        csvPerformanceFile =  open(self.basePath+"\\PerformanceInfo_"+self.id_file_csvPerformance+'.csv', 'ab')
        csvwriter = csv.writer(csvPerformanceFile, delimiter=',',quoting=csv.QUOTE_ALL)
        costFileId = "PerformanceInfo_"+self.id_file_csvPerformance+ "_"+LogId
        csvwriter.writerow([epoch,costFileId,validation_loss,test_score, minibatch_index, iter, best_validation_loss, best_iter, done_looping, patience ])

        f = file(self.basePath + "\\" + costFileId + '.pkl', 'w+b')
        cPickle.dump(cost, f, protocol=2)
        f.close()
        csvPerformanceFile.close()