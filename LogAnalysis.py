__author__ = 'Giovanni'


import cPickle

#fLoaded = file(r'E:\\dev\\TesisTest\\logManager\\PerformanceInfo_Testing_1_4.pkl', 'rb')
fLoaded = file(r'E:\\dev\\TesisTest\\logManager\\Weights_Testing_1_0.pkl', 'rb')
data = cPickle.load(fLoaded)
fLoaded.close()