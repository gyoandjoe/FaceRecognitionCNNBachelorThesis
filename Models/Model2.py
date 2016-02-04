__author__ = 'Giovanni'
import FaceRecognition_CNN
#print theano.config.optimizer


fr = FaceRecognition_CNN.FaceRecognition_CNN(
    ActiveTest=False,
    modeEvaluation = False,
    testFileReference="TestRandReference.csv",
    learning_rate = 0.01, #1e-2 = 0.01
    L2_reg = 0.0005, #5e-4 = 0.0005,
    batch_size = 10, #10 #65 #52
    learningRateFile="E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random_1gb\\LR.txt",
    no_total_rows_in_trainSet=3900*177,
    no_total_rows_in_testSet= 1300*3,
    no_total_rows_in_validationSet=1300*3,
    no_rows_in_train_superBatch=3900,     no_rows_in_test_superBatch=1300,    no_rows_in_validation_superBatch=1300,
    basePathOfReferenceCSVs="E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random_1gb\\",
    basePathOfDataSet = "E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random_1gb\\",
    basePathForLog =  "E:\\dev\\TesisTest\\logManagerTest",
    PerformanceFileId="v0.3_test_TODELETE",
    weightsFileId="v0.3_test_TODELETE",
    randomEveryEpoch=True,
    )



fr.Train(
    patience = 200000,
    n_epochs = 30,
    restoreBackup = False,
    logId='1_0', #1_0
    validation_frequency=5000, #in training batches
    trainigin_info_frequency = 20, #in training batches
    withTestValidation = False,
    backup_frequency=500,

    noItersForUpdateLR = 40
)

print ("OK")
"""
valTest = train_model()
nNoCeros=np.count_nonzero(valTest)
print "Numeros diferente de Cero " + str(nNoCeros)
sumna = np.sum(valTest)
valTest2 = train_model()
print ("...")
"""