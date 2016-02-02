__author__ = 'Giovanni'
import sys
import os
import timeit

import theano
import theano.tensor as T
import numpy as np

from DataAccess.BatchManager import BatchManager
from Arquitecture import FRCNN
import matplotlib.pyplot as plt
import LogManager.LogManager
import time
import time

theano.config.exception_verbosity='high'

class FaceRecognition_CNN(object):

    #los parametros de no_rows_in_train_superBatch, no_rows_in_test_superBatch, no_rows_in_validation_superBatch estan pensados
    #para ocupar entre todos mas de 2GB
    def __init__(self,ActiveTest = True, modeEvaluation = False,testFileReference="TestRandReference.csv", learning_rate = 0.001, pdrop=0.4, L2_reg = 0.0005,batch_size = 20, no_total_rows_in_trainSet=592148, no_total_rows_in_testSet=197382,no_total_rows_in_validationSet=197382,no_rows_in_train_superBatch=15600 ,no_rows_in_test_superBatch=5200,no_rows_in_validation_superBatch=5200,basePathOfReferenceCSVs="E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\",basePathOfDataSet = "E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\", basePathForLog =  "E:\\dev\\TesisTest\\logManager",PerformanceFileId="Testing", weightsFileId="Testing"):

        #self.no_total_rows_in_trainSet = 592148
        #self.no_total_rows_in_testSet = 197382
        #self.no_total_rows_in_validationSet = 197382

        ############################################################################################################
        ############################################# Variables definition #########################################
        ############################################################################################################
        self.no_total_rows_in_trainSet = no_total_rows_in_trainSet
        self.no_total_rows_in_testSet = no_total_rows_in_testSet
        self.no_total_rows_in_validationSet = no_total_rows_in_validationSet

        self.no_rows_in_train_superBatch =  no_rows_in_train_superBatch
        self.no_rows_in_test_superBatch = no_rows_in_test_superBatch
        self.no_rows_in_validation_superBatch = no_rows_in_validation_superBatch

        self.Lm = LogManager.LogManager.LogManager(
            basePath = basePathForLog,
            id_fileWeights =weightsFileId,
            id_fileLossValues="",
            id_file_csvPerformance=PerformanceFileId,
            arquitecture=""
            )
        is_training = T.iscalar('is_training')

        x = T.tensor3('x')  # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        #srng_droput =T.shared_randomstreams.RandomStreams(seed=12345)
        random_droput = np.random.RandomState(1234)
        rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

        self.batch_size = batch_size
        # dimensions are (height, width, channel)
        img_input = x.reshape((self.batch_size, 1, 100, 100))


        self.InitializeDataVariables()

        ############################################################################################################
        ########################################## Load DATASET ####################################################
        ############################################################################################################

        self.LoadDataSet(basePathOfReferenceCSVs,basePathOfDataSet,testFileReference, ActiveTest,modeEvaluation)
        #############################################################################################################
        ########################################### Build MODEL #####################################################
        #############################################################################################################

        self.classifier = FRCNN.FRCNN(
            rng_droput=rng_droput,
            is_training=is_training,
            img_input=img_input,
            noImages=self.batch_size,
            logManager = self.Lm,
            pdrop=pdrop
        )

        if(ActiveTest == True or modeEvaluation == True):
            self.test_model = theano.function(
                inputs=[index],
                outputs=self.classifier.FC.errors(y),
                givens={
                    x: self.bmTestSet.currentX[index * self.batch_size: (index + 1) * self.batch_size,:,:],
                    y: T.cast(self.bmTestSet.currentY[index * self.batch_size: (index + 1) * self.batch_size], 'int32'),
                    is_training: np.cast['int32'](0)
                }
            )

        if modeEvaluation != True:
            self.validate_model = theano.function(
                inputs=[index],
                outputs=self.classifier.FC.errors(y),
                givens={
                    x: self.bmValidationSet.currentX[index * self.batch_size: (index + 1) * self.batch_size,:,:],
                    y: T.cast(self.bmValidationSet.currentY[index * self.batch_size: (index + 1) * self.batch_size], 'int32'),
                    is_training: np.cast['int32'](0)
                }
            )

            # create a list of all model parameters to be fit by gradient descent
            cost = self.classifier.FC.negative_log_likelihood(y) +L2_reg * self.classifier.FC.L2_sqr


            params = self.classifier.FC.params + self.classifier.Conv_51_52.layer0.params + self.classifier.Conv_51_52.layer1.params + self.classifier.Conv_41_42.layer0.params + self.classifier.Conv_41_42.layer1.params + self.classifier.Conv_31_32.layer0.params + self.classifier.Conv_31_32.layer1.params + self.classifier.Conv_21_22.layer0.params + self.classifier.Conv_21_22.layer1.params + self.classifier.Conv_11_12.layer0.params + self.classifier.Conv_11_12.layer1.params

            # create a list of gradients for all model parameters
            grads = T.grad(cost, params)

            # train_model is a function that updates the model parameters by
            # SGD Since this model has many parameters, it would be tedious to
            # manually create an update rule for each model parameter. We thus
            # create the updates list by automatically looping over all
            # (params[i], grads[i]) pairs.
            updates = [
                (param_i, param_i - (learning_rate * grad_i))
                for param_i, grad_i in zip(params, grads)
            ]


            self.train_model = theano.function(
                [index],
                  cost, #self.classifier.FC.p_y_given_x,#dropout.output
                  updates = updates,
                  givens = { x: self.bmTrainSet.currentX[index * self.batch_size: (index + 1) * self.batch_size,:,:],
                    y: T.cast(self.bmTrainSet.currentY[index * self.batch_size: (index + 1) * self.batch_size], 'int32'),
                    is_training: np.cast['int32'](1),
                  }
                  #on_unused_input='warn'
                  #allow_input_downcast=True
            )

    def InitializeDataVariables(self):


        self.minibatchValidation_index_with_offset = 0
        self.minibatchTest_index_with_offset = -1

        self.n_train_batches =  self.no_total_rows_in_trainSet // self.batch_size
        if (self.no_total_rows_in_trainSet % self.batch_size != 0):
            self.n_train_batches = self.n_train_batches + 1

        self.n_test_batches = self.no_total_rows_in_testSet // self.batch_size
        if (self.no_total_rows_in_testSet % self.batch_size != 0):
            self.n_test_batches = self.n_test_batches + 1

        self.n_valid_batches = self.no_total_rows_in_validationSet // self.batch_size
        if (self.no_total_rows_in_validationSet % self.batch_size != 0):
            self.n_valid_batches=self.n_valid_batches + 1

    def LoadDataSet(self,basePathOfReferenceCSVs,basePathOfDataSet,testFileReference,ActiveTest = True,modeEvaluation = False, ):
        if modeEvaluation != True:
            self.bmTrainSet = BatchManager(self.batch_size, self.no_rows_in_train_superBatch, basePathOfReferenceCSVs + "TrainRandReference.csv",basePathOfDataSet+"randTrain")
            self.bmTrainSet.UpdateCurrentXAdYByBatchIndex(0)
            self.bmValidationSet = BatchManager(self.batch_size,self.no_rows_in_validation_superBatch, basePathOfReferenceCSVs + "ValidRandReference.csv",basePathOfDataSet+"randValid")
            self.bmValidationSet.UpdateCurrentXAdYByBatchIndex(0)

        if (ActiveTest == True or modeEvaluation == True):
            self.bmTestSet = BatchManager(self.batch_size,self.no_rows_in_test_superBatch, basePathOfReferenceCSVs + testFileReference,basePathOfDataSet+"randTest")
            self.bmTestSet.UpdateCurrentXAdYByBatchIndex(0)


    def EvaluateModel(self,logId="1_0"):
        print "... Loading Weights"
        restoredValues = self.classifier.GetAndLoadState(logId)
        # test it on the test set
        print "... Testing"
        test_losses = [
            self.do_Test(i)
            for i in xrange(self.n_test_batches)
        ]
        test_score = np.mean(test_losses)
        #logContent = "%s | Looping %d times(iters) took %f seconds(%i minutes), %d examples processed, epoch %i, last cost %f, minibatch %i/%i" % (now, iter+1, seconds_test,seconds_test // 60, iter * self.batch_size, epoch, cost_ij,minibatch_index + 1, self.n_train_batches)
        logContent ='%s | test error: %f ' % (self.Lm.id_file_csvPerformance, (test_score * 100.))
        print(logContent)
        self.Lm.saveLogPerformanceInfo(logContent + "\n")
        #self.Lm.saveTrainingPerformanceInfo(now, epoch, cost_ij,minibatch_index, iter, best_validation_loss, best_iter, done_looping, patience)



    def Train(self,n_epochs = 200,restoreBackup = False, logOfWeightsForLoading="None",fileWeightsForLoading="None", basePathForLoading="None",  logId="1_0",validation_frequency = 100, trainigin_info_frequency = 1000, withTestValidation = True, backup_frequency=20000, patience = 100000):




        ############################################################################################################
        ############################################ TRAINING ######################################################
        ############################################################################################################

        print '... training'
        #    LogManager.LogManager("","","E:\\dev\\TesisTest\\logManager\\testgraph.csv","ok")



            # early-stopping parameters
        #patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.595  #(0 to 1) a relative improvement of this much is
                                           # considered significant
        #validation_frequency = validation_frequency #min(self.n_train_batches, patience / 2)
                                          # go through this many
                                          # minibatche before checking the network
                                          # on the validation set; in this case we
                                          # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.

        epoch = 0
        done_looping = False

        if (restoreBackup == True):
            if (logOfWeightsForLoading != "None"):
                print "Error: only 1 set of weight is permitted"
                return
            #Loading values
            restoredValues = self.classifier.GetAndLoadState(logId)

            # 0 minibatchTrain_index_with_offset,
            # 1 epoch,
            # 2 minibatch_index,
            # 3 best_validation_loss,
            # 4 best_iter,
            # 5 np.cast['int32'](done_looping),
            # 6 patience
            #assign values  to intermediate training values
            restored_minibatchTrain_index_with_offset = restoredValues[0]
            restored_epoch = restoredValues[1]
            restored_minibatch_index =restoredValues[2]

            #assign values directly to  training values
            best_validation_loss=restoredValues[3]
            best_iter=restoredValues[4]
            done_looping = restoredValues[5]
            if patience==-1:
                patience = restoredValues[6]

        if (logOfWeightsForLoading != "None"):
            self.classifier.LoadWeightFromLogId(logOfWeightsForLoading,fileWeightsForLoading, basePathForLoading)

        start_time = timeit.default_timer()
        t_test_start = time.time()

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            minibatchTrain_index_with_offset=0

            if restoreBackup == True:
                if epoch !=  restored_epoch:
                    continue

            if epoch != 0:
                self.classifier.saveState(str(epoch)+"_justStarting",np.asarray([minibatchTrain_index_with_offset,epoch, 0, best_validation_loss, best_iter, np.cast['float64'](0), patience]))

            for minibatch_index in xrange(self.n_train_batches):

                if restoreBackup == True:
                    if minibatch_index !=  restored_minibatch_index:
                        continue


                #No of training batches processed including epochs
                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                self.bmTrainSet.UpdateCurrentXAdYByBatchIndex(minibatch_index)
                if (self.bmTrainSet.dataLoader.IsNewDataSet == True):
                    minibatchTrain_index_with_offset = 0

                if (restoreBackup == True):
                    minibatchTrain_index_with_offset = restored_minibatchTrain_index_with_offset

                if ((iter+1) % backup_frequency == 0) and restoreBackup == False:
                    self.classifier.saveState(str(epoch)+"_"+str(iter),np.asarray([minibatchTrain_index_with_offset,epoch, minibatch_index, best_validation_loss, best_iter, np.cast['float64'](done_looping), patience]))
                    #self.classifier.saveState(str(epoch)+"_"+str(iter),np.asarray([minibatchTrain_index_with_offset,epoch, minibatch_index, best_validation_loss, best_iter, patience]))


                #self.classifier.saveCNNWeights("Before")
                cost_ij = self.train_model(minibatchTrain_index_with_offset)



                #Each 100 iter we print the number of iter
                if (iter+1) % trainigin_info_frequency == 0:
                    t_test_end = time.time()
                    seconds_test = t_test_end - t_test_start
                    now = time.strftime("%c")
                    logContent = "%s | Looping %d times(iters), took %i minutes, %d examples processed, epoch %i, last cost %f, minibatch %i/%i" % (now, iter+1, seconds_test // 60, iter * self.batch_size, epoch, cost_ij,minibatch_index + 1, self.n_train_batches)
                    print(logContent)
                    self.Lm.saveLogPerformanceInfo(logContent + "\n")
                    self.Lm.saveTrainingPerformanceInfo(now, epoch, cost_ij,minibatch_index, iter, best_validation_loss, best_iter, done_looping, patience)



                #Se evalua cada cierto numero de training batches
                if ((iter + 1) % validation_frequency == 0) and restoreBackup == False:
                    print "... Validating"
                    # compute zero-one loss on validation set
                    validation_losses = [self.do_validation(i) for i
                                        in xrange(self.n_valid_batches)]

                    this_validation_loss = np.mean(validation_losses)

                    valInfo = 'epoch %i, minibatch %i/%i, validation error %f %%' %(epoch, minibatch_index + 1, self.n_train_batches, this_validation_loss * 100.)
                    self.Lm.saveLogPerformanceInfo(valInfo + "\n")
                    print(valInfo)


                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        #Si mejora bastante todavia no se ha acercado al minimo global entonces tratamos de incrementar la paciencia,
                        # pero si no mejora bastante significa que el mejoramiento es poco por lo tanto esta muy cerca del minimo
                        # y no necesitamos actualizar la paciencia (que es el numero de batches procesados)
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        if withTestValidation == True:
                            # test it on the test set
                            print "... Testing"
                            test_losses = [
                                #test_model(i)
                                self.do_Test(i)
                                for i in xrange(self.n_test_batches)
                            ]
                            test_score = np.mean(test_losses)
                            print(('     epoch %i, minibatch %i/%i, test error of '
                                       'best model %f %%') %
                                    (epoch, minibatch_index + 1, self.n_train_batches,
                                    test_score * 100.))


                    print "Best Validation loss until now: " + str(best_validation_loss) + " in iter " + str(best_iter)
                    #Se guardan logs de performance
                    self.Lm.savePerformanceInfo(str(epoch)+"_"+str(iter), str(cost_ij),this_validation_loss,test_score,epoch,minibatch_index,iter,best_validation_loss,best_iter,done_looping,patience)


                minibatchTrain_index_with_offset =  minibatchTrain_index_with_offset + 1

                #Como ya todas las asignaciones con respecto de restoreBackup fueron hechas, en la siguiente iteracion ya no es necesario volver a cargar las variables de restore
                if restoreBackup == True:
                    restoreBackup = False

                #Si sobrepasamos o llegamos a la paciencia maxima, detenemos el entrenamiento
                if patience <= iter:
                    done_looping = True
                    break



        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
                  'with test performance %f %%' %
            (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                                os.path.split(__file__)[1] +
                                ' ran for %.2fm' % ((end_time - start_time) / 60.))

    def do_validation(self,indexBatch):
        self.bmValidationSet.UpdateCurrentXAdYByBatchIndex(indexBatch)
        if (self.bmValidationSet.dataLoader.IsNewDataSet == True):
            self.minibatchValidation_index_with_offset = 0
        self.minibatchValidation_index_with_offset = self.minibatchValidation_index_with_offset + 1
        return self.validate_model(self.minibatchValidation_index_with_offset -1)

    def do_Test(self,indexBatch):
        self.bmTestSet.UpdateCurrentXAdYByBatchIndex(indexBatch)
        if (self.bmTestSet.dataLoader.IsNewDataSet == True):
            self.minibatchTest_index_with_offset = 0
        else:
            self.minibatchTest_index_with_offset = self.minibatchTest_index_with_offset + 1
        resultTest = self.test_model(self.minibatchTest_index_with_offset)
        print "Index batch"+ str(indexBatch)+ "result: " + str(resultTest)
        return resultTest



############################################################################
############################## TESTING #####################################
############################################################################
