__author__ = 'Giovanni'
import numpy as np
from sklearn.cross_validation import train_test_split
import csv


def PersistToFile(targetName,data):
    csvfile =  open(targetName, 'ab')
    csvwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONE)
    for item in data:
        csvwriter.writerow([item[0],item[1],item[2]])



dataSetBatchList = np.loadtxt( "E:\My Documents\BUAP\Titulacion\Tesis\Resources\Data Sets\CASIA Processing\Results\\randomFullReasigned.csv",delimiter=',', dtype='str')

trainSet, testValidationSet =  train_test_split(dataSetBatchList, test_size=0.3, random_state=1)
validationSet, testSet = train_test_split(testValidationSet, test_size=0.5, random_state=1)

PersistToFile("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\RANDOM\\trainSet_rand.csv",trainSet)
PersistToFile("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\RANDOM\\validationSet_rand.csv",validationSet)
PersistToFile("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\RANDOM\\testSet_rand.csv",testSet)



print "OK"

