__author__ = 'Giovanni'

import DataAccess.DataLoader
da = DataAccess.DataLoader.DataLoader("E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\Distribute and random_1gb\TrainRandReference.csv","E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random_1gb\\randTrain\\" )

da1 = da.getDataSetByBatchIndex(0)

da2 = da.getDataSetByBatchIndex(1)

da3 = da.getDataSetByBatchIndex(2)

da3 = da.getDataSetByBatchIndex(100)

print "OK"