__author__ = 'Giovanni'
import numpy as np
from PIL import Image

import cPickle

class PreprocessingCASIA(object):
    def __init__(self,fileReferece,basePath):
        self.fileReferenceData = open(fileReferece,'r')
        self.basePath = basePath
        self.dataList = np.loadtxt(self.fileReferenceData,delimiter=',', dtype='str',usecols = (0,1,2))


    def getBDPart(self, startIndex, sizeBatch):
        DataX = np.zeros((sizeBatch , 100,100),dtype=np.float64)
        DataY = np.zeros((sizeBatch),dtype=np.float64)
        contador=0
        for fileInfo in self.dataList[startIndex:(startIndex + sizeBatch),:]:
            DataX[contador] = np.asarray(Image.open(self.basePath + fileInfo[1] + '\\' + fileInfo[2]), dtype=np.float64) / 256
            DataY[contador] = fileInfo[0]
            contador += 1
            #print fileInfo[0] + " " + fileInfo[2]
        return (DataX,DataY)



#objPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\CASIAFULL.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\')
objPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\CASIAFULL_RANDOM.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\')

print ("List loaded")




startIndex=12
bigBatchSize = 26000
for index in xrange(16):
    if index < startIndex:
        continue
    print 'PROCESSING -- Images to Numpy Array, Index : ' + str(index)
    query = objPreprocessing.getBDPart(index * bigBatchSize, bigBatchSize)
    print '.............. OK: ' + str(index)
    print 'PROCESSING -- Serialize Numpy Array to File, Index : ' + str(index)
    f = file('E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\RANDOM\\CASIAFULL_R_'+ str(index) + '.pkl', 'w+b')
    cPickle.dump(query, f, protocol=2)
    #myf = open("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\RANDOM\\CASIAFULL_R_FULL.pkl","ab+")
    #tests= cPickle.dumps(query,  protocol=2)
    #myf.write(tests)
    #myf.close()
    f.close()
    print '.............. OK: ' + str(index)

#myf.close()
#print "SALTO"
#querytest2 = objPreprocessing.getBDPart(200,400)

#querytest = objPreprocessing.getBDPart(0,986912)


#fLoaded = file('E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\CASIAFULL.pkl', 'rb')
#setloaded = cPickle.load(fLoaded)
#fLoaded.close()

print ("Serialize DataSet -- OK")