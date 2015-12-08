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

theano.config.exception_verbosity='high'

class FaceRecognition_CNN(object):

    def __init__(self, learning_rate = 0.01,pbatch_size = 15):
        ############################################################################################################
        ############################################# Variables definition #########################################
        ############################################################################################################

        self.Lm = LogManager.LogManager.LogManager("","","E:\\dev\\TesisTest\\logManager\\testgraph.csv","ok")
        is_training = T.iscalar('is_training')

        x = T.tensor3('x')  # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        #srng_droput =T.shared_randomstreams.RandomStreams(seed=12345)
        random_droput = np.random.RandomState(1234)
        rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

        self.batch_size = pbatch_size
        # dimensions are (height, width, channel)
        img_input = x.reshape((self.batch_size, 1, 100, 100))

        self.InitializeDataVariables()

        ############################################################################################################
        ########################################## Load DATASET ####################################################
        ############################################################################################################
        self.LoadDataSet()
        #############################################################################################################
        ########################################### Build MODEL #####################################################
        #############################################################################################################

        classifier = FRCNN.FRCNN(
            rng_droput=rng_droput,
            is_training=is_training,
            img_input=img_input,
            noImages=self.batch_size,
            pdrop=0.4
        )
        cost = classifier.FC.negative_log_likelihood(y)

        self.test_model = theano.function(
            inputs=[index],
            outputs=classifier.FC.errors(y),
            givens={
                x: self.bmTestSet.currentX[index * self.batch_size: (index + 1) * self.batch_size,:,:],
                y: T.cast(self.bmTestSet.currentY[index * self.batch_size: (index + 1) * self.batch_size], 'int32'),
                is_training: np.cast['int32'](0)
            }
        )

        self.validate_model = theano.function(
            inputs=[index],
            outputs=classifier.FC.errors(y),
            givens={
                x: self.bmValidationSet.currentX[index * self.batch_size: (index + 1) * self.batch_size,:,:],
                y: T.cast(self.bmValidationSet.currentY[index * self.batch_size: (index + 1) * self.batch_size], 'int32'),
                is_training: np.cast['int32'](0)
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        params = classifier.FC.params + classifier.Conv_51_52.layer0.params + classifier.Conv_51_52.layer1.params + classifier.Conv_41_42.layer0.params + classifier.Conv_41_42.layer1.params + classifier.Conv_31_32.layer0.params + classifier.Conv_31_32.layer1.params + classifier.Conv_21_22.layer0.params + classifier.Conv_21_22.layer1.params + classifier.Conv_11_12.layer0.params + classifier.Conv_11_12.layer1.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        #img = Image.open(r'E:\dev\TesisFRwithCNN\TestRS\0000117\005-l.jpg')
        #img = np.asarray(img, dtype=theano.config.floatX) / 256


        #Use the Theano flag 'exception_verbosity=high'

        self.train_model = theano.function(
            [index],
              classifier.FC.p_y_given_x,#dropout.output
              updates = updates,
              givens = {
                x: self.bmTrainSet.currentX[index * self.batch_size: (index + 1) * self.batch_size,:,:],
                y: T.cast(self.bmTrainSet.currentY[index * self.batch_size: (index + 1) * self.batch_size], 'int32'),
                is_training: np.cast['int32'](1),
              }
              #on_unused_input='warn'
              #allow_input_downcast=True
        )

    def InitializeDataVariables(self):
        #self.no_total_rows_in_trainSet = 592148
        #self.no_total_rows_in_testSet = 197382
        #self.no_total_rows_in_validationSet = 197382
        self.no_total_rows_in_trainSet = 7800
        self.no_total_rows_in_testSet = 2600
        self.no_total_rows_in_validationSet = 2600

        self.no_rows_in_train_superBatch =  7800
        self.no_rows_in_test_superBatch = 2600
        self.no_rows_in_validation_superBatch = 2600

        self.minibatchValidation_index_with_offset = 0
        self.minibatchTest_index_with_offset = 0


        self.n_train_batches =  self.no_total_rows_in_trainSet // self.batch_size
        if (self.no_total_rows_in_trainSet % self.batch_size != 0):
            self.n_train_batches = self.n_train_batches + 1

        self.n_test_batches = self.no_total_rows_in_testSet // self.batch_size
        if (self.no_total_rows_in_testSet % self.batch_size != 0):
            self.n_test_batches = self.n_test_batches + 1

        self.n_valid_batches = self.no_total_rows_in_validationSet // self.batch_size
        if (self.no_total_rows_in_validationSet % self.batch_size != 0):
            self.n_valid_batches=self.n_valid_batches + 1

    def LoadDataSet(self,basePathOfReferenceCSVs="E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\",basePathOfDataSet = "E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\"):

        self.bmTrainSet = BatchManager(self.batch_size, self.no_rows_in_train_superBatch, basePathOfReferenceCSVs + "TrainRandReference.csv",basePathOfDataSet+"randTrain")
        self.bmTrainSet.UpdateCurrentXAdYByBatchIndex(0)
        self.bmTestSet = BatchManager(self.batch_size,self.no_rows_in_test_superBatch, basePathOfReferenceCSVs + "TestRandReference.csv",basePathOfDataSet+"randTest")
        self.bmTestSet.UpdateCurrentXAdYByBatchIndex(0)
        self.bmValidationSet = BatchManager(self.batch_size,self.no_rows_in_validation_superBatch, basePathOfReferenceCSVs + "ValidRandReference.csv",basePathOfDataSet+"randValid")
        self.bmValidationSet.UpdateCurrentXAdYByBatchIndex(0)

    def Train(self,restoreBackup = False):

        createBackup = False
        if (restoreBackup == True):
            restored_epoch = 0
            restored_minibatch_index =0
            restored_best_validation_loss=0
            restored_best_iter=0
            restored_done_looping=0
            restored_patience=0

        ############################################################################################################
        ############################################ TRAINING ######################################################
        ############################################################################################################

        print '... training'
        #    LogManager.LogManager("","","E:\\dev\\TesisTest\\logManager\\testgraph.csv","ok")


        n_epochs = 200
            # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  #(0 to 1) a relative improvement of this much is
                                           # considered significant
        validation_frequency = 1 #min(self.n_train_batches, patience / 2)
                                          # go through this many
                                          # minibatche before checking the network
                                          # on the validation set; in this case we
                                          # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.

        epoch = 0
        done_looping = False

        start_time = timeit.default_timer()


        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            minibatchTrain_index_with_offset=0

            if restoreBackup == True:
                if epoch !=  restored_epoch:
                    continue

            for minibatch_index in xrange(self.n_train_batches):

                if restoreBackup == True:
                    if minibatch_index !=  restored_minibatch_index:
                        continue

                #No of training batches processed including epochs
                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                #Each 100 iter we print the number of iter
                if iter % 100 == 0:
                    print 'training @ iter = ', iter

                self.bmTrainSet.UpdateCurrentXAdYByBatchIndex(minibatch_index)
                if (self.bmTrainSet.dataLoader.IsNewDataSet == True):
                    minibatchTrain_index_with_offset = 0
                    #if (createBackup == True):
                        #save epoch, minibatch_index, best_validation_loss, best_iter, done_looping, patience

                cost_ij = self.train_model(minibatchTrain_index_with_offset)

                #Se evalua cada cierto numero de training batches
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [self.do_validation(i) for i
                                        in xrange(self.n_valid_batches)]

                    this_validation_loss = np.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                            (epoch, minibatch_index + 1, self.n_train_batches,
                            this_validation_loss * 100.))


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

                        # test it on the test set
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

                    #Se guardan logs de performance

                    self.Lm.savePerformanceData(cost_ij,this_validation_loss,test_score,epoch,minibatch_index,iter,best_validation_loss,best_iter,done_looping,patience )


                minibatchTrain_index_with_offset =  minibatchTrain_index_with_offset + 1

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
        self.minibatchTest_index_with_offset = self.minibatchTest_index_with_offset + 1
        return self.test_model(self.minibatchTest_index_with_offset -1)

""""
if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
"""

############################################################################
############################## TESTING #####################################
############################################################################


fr = FaceRecognition_CNN()
fr.Train()
print ("OK")
"""
valTest = train_model()
nNoCeros=np.count_nonzero(valTest)
print "Numeros diferente de Cero " + str(nNoCeros)
sumna = np.sum(valTest)
valTest2 = train_model()
print ("...")
"""