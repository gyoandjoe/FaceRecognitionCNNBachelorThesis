import random

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


file = open("newfile.txt", "w")
file.write("0.000159")
file.close()

file2 = open('newfile.txt', 'r')
varrecupered = float(file2.read()) #np.cast['float'](1)(file.read())
file2.close()
varrecuperedassiged = varrecupered + 5.1

items = range(10)
resulrand1 = random.shuffle(items)
resulrand2 = random.shuffle(items)
resulrand3 = random.shuffle(items)

############################################################################
############################# Variables definition #########################
############################################################################
is_training = T.iscalar('is_training')
x = T.tensor3('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels
# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
batch_size = 10
learning_rate = 0.01
n_epochs = 200
# dimensions are (height, width, channel)
img_input = x.reshape((batch_size, 1, 100, 100))

#srng_droput =T.shared_randomstreams.RandomStreams(seed=12345)
random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))
mask=rng_droput.binomial(n=1,size=(10,10),p=1, dtype=theano.config.floatX)
dropTestF = theano.function([],mask)
dropEval = dropTestF()

n_train_batches = 592148 // batch_size
n_test_batches = 197382 // batch_size
n_valid_batches = 197382 // batch_size
############################################################################
########################## Load DATASET ####################################
############################################################################

bmTrainSet = BatchManager(batch_size,7800,"E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\TrainRandReference.csv","E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\randTrain")
bmTrainSet.UpdateCurrentXAndY(0)
bmTestSet = BatchManager(batch_size,2600,"E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\TestRandReference.csv","E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\randTest")
bmTestSet.UpdateCurrentXAndY(0)
bmValidationSet = BatchManager(batch_size,2600,"E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\ValidRandReference.csv","E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\randValid")
bmValidationSet.UpdateCurrentXAndY(0)


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


# create a list of all model parameters to be fit by gradient descent
#params = classifier.Conv_51_52.layer0.params #classifier.FC.params #+ classifier.Conv_51_52.layer0.params #+ classifier.Conv_51_52.layer1.params + classifier.Conv_41_42.layer0.params + classifier.Conv_41_42.layer1.params + classifier.Conv_31_32.layer0.params + classifier.Conv_31_32.layer1.params + classifier.Conv_21_22.layer0.params + classifier.Conv_21_22.layer1.params + classifier.Conv_11_12.layer0.params + classifier.Conv_11_12.layer1.params
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
      classifier.FC.negative_log_likelihood(y), # p_y_given_x, #seeY(y), #negative_log_likelihood(y),#dropout.output
      updates = updates,
      givens = {
        x: bmTrainSet.currentX[index * batch_size: (index + 1) * batch_size,:,:],
        #y: T.cast(bmTrainSet.currentY[index * batch_size: (index + 1) * batch_size], 'int32'),
        y: T.cast(bmTrainSet.currentY[index * batch_size: (index + 1) * batch_size], 'int32'),
        is_training: np.cast['int32'](1)

      },
      on_unused_input='warn'
      #allow_input_downcast=True
)

dataXNp0 = train_model(0)


print ("OK")