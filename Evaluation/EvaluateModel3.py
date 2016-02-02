__author__ = 'Giovanni'
import Models.FaceRecognition_CNN
#print theano.config.optimizer


fr = Models.FaceRecognition_CNN.FaceRecognition_CNN(
    ActiveTest=False,
    modeEvaluation = True,
    testFileReference="TrainRandReference.csv",#"TestRandReference.csv",
    learning_rate = 0.01, #1e-2 = 0.01
    L2_reg = 0.0005, #5e-4 = 0.0005
    batch_size = 10,
    no_total_rows_in_trainSet=3900*177,
    no_total_rows_in_testSet= 3900 * 20,#1300 * 5,
    no_total_rows_in_validationSet=1300 * 3,
    no_rows_in_train_superBatch=3900,
    no_rows_in_test_superBatch=3900, #1300,
    no_rows_in_validation_superBatch=1300,
    basePathOfReferenceCSVs="E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random_1gb\\",
    basePathOfDataSet = "E:\\My Documents\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random_1gb\\",
    basePathForLog =  "E:\\dev\\TesisTest\\logModel3",
    #basePathForLog =  "E:\\dev\\TesisTest\\logManager",
    #PerformanceFileId="v0.1_test_TODELETE_eval",
    PerformanceFileId="Model3Remote",
    #weightsFileId="v0.1_test_TODELETE"
    weightsFileId="model3Experiment"
    )

fr.EvaluateModel(
    logId= "7_72999"
)