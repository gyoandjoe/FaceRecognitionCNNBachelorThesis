

__author__ = 'Giovanni'
import sys
import os
import timeit

import theano
import theano.tensor as T
import numpy as np

from DataAccess.BatchManager import BatchManager
from Arquitecture import FRCNN

theano.config.exception_verbosity='high'
############################################################################
############################# Variables definition #########################
############################################################################
is_training = T.iscalar('is_training')
x = T.tensor3('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels
# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
batch_size = 10
learning_rate = 00.1
n_epochs = 200
# dimensions are (height, width, channel)
img_input = x.reshape((batch_size, 1, 100, 100))

#srng_droput =T.shared_randomstreams.RandomStreams(seed=12345)
random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

n_train_batches = 592148 // batch_size
n_test_batches = 197382 // batch_size
n_valid_batches = 197382 // batch_size
############################################################################
########################## Load DATASET ####################################
############################################################################

bmTrainSet = BatchManager(batch_size,7800,"E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\TrainRandReference.csv","E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\randTrain")
bmTrainSet.UpdateCurrentXAdY(0)
bmTestSet = BatchManager(batch_size,2600,"E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\TestRandReference.csv","E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\randTest")
bmTestSet.UpdateCurrentXAdY(0)
bmValidationSet = BatchManager(batch_size,2600,"E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\ValidRandReference.csv","E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\randValid")
bmValidationSet.UpdateCurrentXAdY(0)


#############################################################################
########################### Build MODEL #####################################
#############################################################################

classifier = FRCNN.FRCNN(
    rng_droput=rng_droput,
    is_training=is_training,
    img_input=img_input,
    noImages=batch_size,
    pdrop=0.4
)
cost = classifier.FC.negative_log_likelihood(y)


test_model = theano.function(
    inputs=[index],
    outputs=classifier.FC.errors(y),
    givens={
        x: bmTestSet.currentX[index * batch_size: (index + 1) * batch_size,:,:],
        y: T.cast(bmTestSet.currentY[index * batch_size: (index + 1) * batch_size], 'int32'),
        is_training: np.cast['int32'](0)
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=classifier.FC.errors(y),
    givens={
        x: bmValidationSet.currentX[index * batch_size: (index + 1) * batch_size,:,:],
        y: T.cast(bmValidationSet.currentY[index * batch_size: (index + 1) * batch_size], 'int32'),
        is_training: np.cast['int32'](0)
    }
)

# create a list of all model parameters to be fit by gradient descent
params = classifier.FC.params + classifier.Conv_51_52.layer0.params #+ classifier.Conv_51_52.layer1.params + classifier.Conv_41_42.layer0.params + classifier.Conv_41_42.layer1.params + classifier.Conv_31_32.layer0.params + classifier.Conv_31_32.layer1.params + classifier.Conv_21_22.layer0.params + classifier.Conv_21_22.layer1.params + classifier.Conv_11_12.layer0.params + classifier.Conv_11_12.layer1.params

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

train_model = theano.function(
    [index],
      classifier.FC.p_y_given_x,#dropout.output
      updates = updates,
      givens = {
        x: bmTrainSet.currentX[index * batch_size: (index + 1) * batch_size,:,:],
        y: T.cast(bmTrainSet.currentY[index * batch_size: (index + 1) * batch_size], 'int32'),
        is_training: np.cast['int32'](1),
      }
      #on_unused_input='warn'
      #allow_input_downcast=True
)

dataXNp0 = train_model(0)



############################################################################
############################ TRAINING ######################################
############################################################################

print '... training'
    # early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                        # found
improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        iter = (epoch - 1) * n_train_batches + minibatch_index


        if iter % 100 == 0:
            print 'training @ iter = ', iter
        cost_ij = train_model(minibatch_index)

        if (iter + 1) % validation_frequency == 0:

            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                    improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [
                    #test_model(i)
                    test_model(bmTestSet.getBatchByIndex(i))
                    for i in xrange(n_test_batches)
                ]
                test_score = np.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                        test_score * 100.))

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

""""
if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
"""

############################################################################
############################## TESTING #####################################
############################################################################

valTest = train_model()

nNoCeros=np.count_nonzero(valTest)

print "Numeros diferente de Cero " + str(nNoCeros)


sumna = np.sum(valTest)

valTest2 = train_model()


print ("...")