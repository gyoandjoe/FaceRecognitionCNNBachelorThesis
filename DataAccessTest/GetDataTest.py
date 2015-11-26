__author__ = 'Giovanni'
import DataAccess.DataLoader
import DataAccess.LoadDataFunctions

accessData = DataAccess.DataLoader.DataLoader("E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\datasetTest.csv","E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\RANDOM")
data1 = accessData.getDataSetByBatchIndex(13)

data2 = accessData.getDataSetByBatchIndex(13)







print "OK"