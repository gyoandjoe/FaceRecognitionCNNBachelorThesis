__author__ = 'Giovanni'


import cPickle

#fLoaded = file(r'E:\\dev\\TesisTest\\logManager\\PerformanceInfo_Testing_1_4.pkl', 'rb')
fLoaded = file(r'E:\\dev\\TesisTest\\logManager\\PerformanceInfo_Testing_8_17999.pkl', 'rb')
data = cPickle.load(fLoaded)
BestValidation = data[6]
fLoaded.close()