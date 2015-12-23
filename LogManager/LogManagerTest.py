__author__ = 'Giovanni'

import LogManager

Lm= LogManager.LogManager("","","E:\\dev\\TesisTest\\logManager\\testgraph.csv","ok")
Lm.savePerformanceInfo(0,1,2,3,4,"ITER",5,6,7,"patiente" )
Lm.savePerformanceInfo(8,9,10,11,12,"ITER2",13,14,15,"patiente2" )
print ("OK")