__author__ = 'Giovanni'
import csv


class LogManager(object):
    def __init__(self, fileWights, fileLossValues, csvPerformanceFileName, arquitecture):
        self.csvPerformanceFileName = csvPerformanceFileName
        self.fileWeights = fileWights
        self.fileLossValues = fileLossValues
    def saveState_Weights(self):
        return 0

    def saveState_TrainProcessData(self, epoch, minibatch_index, best_validation_loss, best_iter, done_looping, patience):
        return 0

    def savePerformanceData(self,cost,validation_loss,test_score,epoch, minibatch_index, iter, best_validation_loss, best_iter, done_looping, patience):
        csvPerformanceFile =  open(self.csvPerformanceFileName, 'ab')
        csvwriter = csv.writer(csvPerformanceFile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([epoch, cost,validation_loss,test_score,epoch, minibatch_index, iter, best_validation_loss, best_iter, done_looping, patience ])
        csvPerformanceFile.close()