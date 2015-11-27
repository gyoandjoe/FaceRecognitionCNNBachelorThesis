__author__ = 'Giovanni'
import numpy as np
from PIL import Image

import cPickle

class PreprocessingCASIA(object):
    def __init__(self,fileReferece,basePath,bigBatchSize,prefixFullFile):
        self.fileReferenceData = open(fileReferece,'r')
        self.basePath = basePath
        self.dataList = np.loadtxt(self.fileReferenceData,delimiter=',', dtype='str',usecols = (0,1,2))
        self.bigBatchSize = bigBatchSize
        self.prefixFullFile = prefixFullFile

    def getBDPart(self, startIndex, sizeBatch):
        contador=0

        data=  self.dataList[startIndex:(startIndex + sizeBatch),:]

        DataX = np.zeros((data.shape[0] , 100,100),dtype=np.float64)
        DataY = np.zeros((data.shape[0]),dtype=np.float64)

        for fileInfo in data:
            DataX[contador] = np.asarray(Image.open(self.basePath + fileInfo[1] + '\\' + fileInfo[2]), dtype=np.float64) / 256
            DataY[contador] = fileInfo[0]
            contador += 1
        return (DataX,DataY)


    def StartProcess(self, startIndex,endIndex):
        for index in xrange(endIndex):
            if index < startIndex:
                continue
            print 'PROCESSING -- Images to Numpy Array, Index : ' + str(index)
            query = self.getBDPart(index * self.bigBatchSize, self.bigBatchSize)
            print '.............. OK: ' + str(index)
            print 'PROCESSING -- Serialize Numpy Array to File, Index : ' + str(index)
            f = file(self.prefixFullFile + str(index) + '.pkl', 'w+b')
            cPickle.dump(query, f, protocol=2)
            f.close()
            print '.............. OK: ' + str(index)
        print ("Serialize DataSet -- OK")

#objPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\CASIAFULL.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\')


    #myf = open("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\RANDOM\\CASIAFULL_R_FULL.pkl","ab+")
    #tests= cPickle.dumps(query,  protocol=2)
    #myf.write(tests)
    #myf.close()


#myf.close()
#print "SALTO"
#querytest2 = objPreprocessing.getBDPart(200,400)

#querytest = objPreprocessing.getBDPart(0,986912)


#fLoaded = file('E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\CASIAFULL.pkl', 'rb')
#setloaded = cPickle.load(fLoaded)
#fLoaded.close()

