__author__ = 'Giovanni'

import FRCNN
import theano
import theano.tensor as T
from PIL import Image
import numpy as np

is_training = T.iscalar('is_training')
x = T.matrix('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels
# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
batch_size = 500
learning_rate = 0.1
n_epochs = 200


# dimensions are (height, width, channel)
img_input = x.reshape((1, 1, 100, 100))


#srng_droput =T.shared_randomstreams.RandomStreams(seed=12345)
random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

classifier = FRCNN.FRCNN(
    rng_droput=rng_droput,
    is_training=is_training,
    img_input=img_input,
    pdrop=0.4
)
cost = classifier.negative_log_likelihood(y)

test_model = theano.function(
    inputs=[index],
    outputs=classifier.FC.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size],
        is_training: np.cast['int32'](0)
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=classifier.FC.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size],
        is_training: np.cast['int32'](0)
    }
)

# create a list of all model parameters to be fit by gradient descent
params = classifier.FC.params + classifier.Conv_51_52.layer0.params + classifier.Conv_51_52.layer1.params + classifier.Conv_41_42.layer0.params + classifier.Conv_41_42.layer1.params + classifier.Conv_31_32.layer0.params + classifier.Conv_31_32.layer1.params + classifier.Conv_51_52.layer0.params + classifier.Conv_21_22.layer1.params + classifier.Conv_11_12.layer0.params + classifier.Conv_11_12.layer1.params

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

img = Image.open(r'E:\dev\TesisFRwithCNN\TestRS\0000117\005-l.jpg')
img = np.asarray(img, dtype=theano.config.floatX) / 256


train_model = theano.function(
    [],
      classifier.FC.p_y_given_x,#dropout.output
      updates = updates,
      givens = {
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
#       x: img,
        is_training: np.cast['int32'](1)
      },
      on_unused_input='warn'
)

valTest = train_model()

nNoCeros=np.count_nonzero(valTest)

print "Numeros diferente de Cero " + str(nNoCeros)


sumna = np.sum(valTest)

valTest2 = train_model()


print ("...")