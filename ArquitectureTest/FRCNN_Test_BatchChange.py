__author__ = 'Giovanni'
"""
Se pretende probar el cambio de carga de superbatch del dataset
cuando se recorren varios trainBatches
"""

import theano
import theano.tensor as T
import numpy as np

from DataAccess.BatchManager import BatchManager
from Arquitecture import FRCNN
v = theano.__version__
batch_size = 100
is_training = T.iscalar('is_training')

index = T.lscalar()



x = T.tensor3('x')  # the data is presented as rasterized images ok

#xx = T.matrix('xx',dtype=theano.config.floatX)
y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels
#yy = T.ivector('yy')

img_input = x.reshape((batch_size, 1, 100, 100))

random_droput = np.random.RandomState(1234)
rng_droput = T.shared_randomstreams.RandomStreams(random_droput.randint(999999))

bmTrainSet = BatchManager(batch_size,7800,"E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\TrainRandReference.csv","E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\Distribute and random\\randTrain")


xfl = T.tensor4('xfl')
classifier = FRCNN.FRCNN(
    rng_droput=rng_droput,
    is_training=is_training,
    img_input= img_input, #xfl,
    noImages=batch_size,
    pdrop=0.4
)
theano.config.exception_verbosity='high'

#dataXNp = np.asarray(bmTrainSet.getRawDataSet(0)[0],dtype=theano.config.floatX)
#dataYNp = np.asarray(bmTrainSet.getRawDataSet(0)[1],dtype=theano.config.floatX)

#ssh = theano.shared(dataXNp,borrow=True)

bmTrainSet.UpdateCurrentXAdYByBatchIndex(0)

train_models1 = theano.function(
    [index],
    classifier.FC.seeY(y),
    rebuild_strict =False,
    givens = {
        x: bmTrainSet.currentX[index * batch_size: (index + 1) * batch_size,:,:],
        #x: ssh[index * batch_size: (index + 1) * batch_size,:,:],
        y: T.cast(bmTrainSet.currentY[index * batch_size: (index + 1) * batch_size], 'int32'),
        is_training: np.cast['int32'](0)
    },
    on_unused_input='warn'
)
dataXNp0 = train_models1(0)
dataXNp1 = train_models1(1)
dataXNp1_1 = train_models1(2)
bmTrainSet.UpdateCurrentXAdYByBatchIndex(77)
if (bmTrainSet.dataLoader.IsNewDataSet == True):
    print "Nuevos datos"
else:
    print "Los mismos datos"

dataXNp2 = train_models1(0)
dataXNp3 = train_models1(1)
dataXNp4 = train_models1(2)
print ("End")
"""
dataXNp = train_models1(0)
dataXNp = train_models1(1)
print "------------------------"
print dataXNp
ssh.set_value(np.asarray(bmTrainSet.getRawDataSet(1)[0],dtype=theano.config.floatX),borrow=True)
dataXNp = train_models1(0)
dataXNp = train_models1(1)
print "------------------------"
print dataXNp
ssh.set_value(np.asarray(bmTrainSet.getRawDataSet(0)[0],dtype=theano.config.floatX), borrow=True)
dataXNp = train_models1(0)
dataXNp = train_models1(1)
print "------------------------"
print dataXNp
######################################################################
"""""

""""
class simple(object):
    def __init__(self, entrada):
        self.entrada=entrada

sim = simple(
    entrada=img_input
)

dataToPassX= T.tensor3('datatopass')
rawData= T.tensor3('rawData')



###################################################################################



dataToPassXS,dataToPassYS=bmTrainSet.getBatchByIndex(0)[0]

#Xfl_update = (dataToPassX, dataToPassXS)
fl = theano.function(
    [dataToPassX],
    sim.entrada,
    givens={
        x: dataToPassX
    }
    ,on_unused_input='warn')

#sentrada = fl(dataXNp)

train_model_fk = theano.function(
    [index],
    classifier.img_input,
    rebuild_strict =True,
    #updates = updates,
    givens = {
        xfl: sim.entrada[index * 10: (index + 1) * 10,:,:,:],
        #y: yy[index * batch_size: (index + 1) * batch_size],
        is_training: np.cast['int32'](0)
    },

    on_unused_input='warn'
)
print ('Ok Model')
resfl1 = fl(dataXNp)

dataXNp = np.asarray(bmTrainSet.getRawDataSet(1)[0],dtype=theano.config.floatX)
resfl2 = fl(dataXNp)

#############################################################################################################
datos=T.tensor3('datos')


train_model_1 = theano.function(
    [index],
    classifier.img_input, #classifier.dropout.output,#classifier.FC.p_y_given_x,
    rebuild_strict =False,
    #updates = updates,
    givens = {
        x: dataToPassX[index * batch_size: (index + 1) * batch_size,:,:],
        y: yy[index * batch_size: (index + 1) * batch_size],
        is_training: np.cast['int32'](0)
    },
    on_unused_input='warn'
    #mode='DebugMode'
)

xx,yy=bmTrainSet.getBatchByIndex(0)
#yy=bmTrainSet.getBatchByIndex(0)[1]
#mmt=xx.get_value()[0:10,:,:]


datos=T.tensor3('datos')


train_model_1 = theano.function(
    [index],
    classifier.img_input, #classifier.dropout.output,#classifier.FC.p_y_given_x,
    rebuild_strict =False,
    #updates = updates,
    givens = {
        x: dataToPassX[index * batch_size: (index + 1) * batch_size,:,:],
        y: yy[index * batch_size: (index + 1) * batch_size],
        is_training: np.cast['int32'](0)
    },
    on_unused_input='warn'
    #mode='DebugMode'
)

#X = shared(numpy.array([0,1,2,3,4], dtype='int'))
#Y = T.lvector()
no=T.iscalar()
#newDAta = bmTrainSet.getBatchByIndex(2700)[0]

#X_update = (xx, bmTrainSet.getBatchByIndex(no)[0])
#f = theano.function(inputs=[no], updates=[X_update],on_unused_input='warn')

res1 = train_model_1(0)




#f(1500)
#bmTrainSet.change()
res1 = train_model_1(0)




tt = np.asarray(bmTrainSet.getRawDataSet(0)[0],dtype=theano.config.floatX)


datos=T.tensor3('datos')

train_model = theano.function(
    [index,datos],
    classifier.FC.p_y_given_x, #classifier.dropout.output,
    rebuild_strict =False,
    #updates = updates,
    givens = {
        x:datos, #theano.shared(datos,borrow=True),
        #y:bmTrainSet.getBatchByIndex(0)[1],
        #x: xx[index * batch_size: (index + 1) * batch_size,:,:],
        y: yy[index * batch_size: (index + 1) * batch_size],
        is_training: np.cast['int32'](1)
    },
    on_unused_input='warn'
    #allow_input_downcast=True #    mode='DebugMode'
)

test1 = train_model(0,tt[0:10,:,:])

test2 = train_model(0,tt[0:10,:,:])

print("OK")

"""""""""""""""