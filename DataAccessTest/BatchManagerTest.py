__author__ = 'Giovanni'
import DataAccess.BatchManager
import theano
import theano.tensor as T

bm = DataAccess.BatchManager.BatchManager(2000,26000,"E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\datasetTest.csv","E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\RANDOM")

##num = T.lscalar()
##ff = theano.function(num,bm.getBatchByIndex(num))


data1 = bm.getBatchByIndex(0)
data2 = bm.getBatchByIndex(1)
data2 = bm.getBatchByIndex(2)
data2 = bm.getBatchByIndex(3)
data3 = bm.getBatchByIndex(12) #aqui pido por 13 a la vez, asi que esta es la ulima vez que socilita el superbatch en index 0
data4 = bm.getBatchByIndex(13) #aqui pido por 14 a la vez, asi que esta es la primera vez que socilita el superbatch en index 1
data5 = bm.getBatchByIndex(14)
data5 = bm.getBatchByIndex(15)
data5 = bm.getBatchByIndex(22)
print "Finish Test"
