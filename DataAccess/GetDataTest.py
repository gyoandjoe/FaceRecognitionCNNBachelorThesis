__author__ = 'Giovanni'
import DataLoader
import LoadDataFunctions

accessData = DataLoader.DataLoader("E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\datasetTest.csv","E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\RANDOM")
data1 = accessData.getDataSetByBatchIndex(13)

data2 = accessData.getDataSetByBatchIndex(13)







print "OK"