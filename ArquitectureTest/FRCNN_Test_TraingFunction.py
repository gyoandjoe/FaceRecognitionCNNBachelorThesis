__author__ = 'Giovanni'
"""
Se pretende probar la funcion de entrenamiento a traves de todas las capas de la arquitectura
y todas los metodos que se usaran en el entrenamiento
"""

import theano
import theano.tensor as T
import numpy as np

from DataAccess.BatchManager import BatchManager
from Arquitecture import FRCNN

batch_size = 10
is_training = T.iscalar('is_training')

index = T.lscalar()

x = T.tensor3('x')  # the data is presented as rasterized images
#xx = T.matrix('xx',dtype=theano.config.floatX)
y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels
#yy = T.ivector('yy')

img_input = x.reshape((batch_size, 1, 100, 100))

random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

bmTrainSet = BatchManager(batch_size,26000,"E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\datasetTest.csv","E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\RANDOM")


classifier = FRCNN.FRCNN(
    rng_droput=rng_droput,
    is_training=is_training,
    img_input=img_input,
    noImages=batch_size,
    pdrop=0.4
)

xx,yy=bmTrainSet.getBatchByIndex(0)
#yy=bmTrainSet.getBatchByIndex(0)[1]
#mmt=xx.get_value()[0:10,:,:]
train_model = theano.function(
    [index],
    classifier.FC.p_y_given_x, #classifier.dropout.output,
    #updates = updates,
    givens = {
        x: xx[index * batch_size: (index + 1) * batch_size,:,:],
        y: yy[index * batch_size: (index + 1) * batch_size],
        is_training: np.cast['int32'](1)
    },
    on_unused_input='warn'
    #allow_input_downcast=True #    mode='DebugMode'
)

mm = train_model(2)

print("OK")