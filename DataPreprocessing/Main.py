from DataPreprocessing.PreprocessingCASIA import PreprocessingCASIA

__author__ = 'Giovanni'

#objPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\trainSet.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',7800,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\Train\\CASIA_TrainSet_NoRandom_')




def createValidationSet():
    #validationPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\validationSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',2600,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\randValid\\CASIA_ValidationSet_Rand_')
    validationPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\validationSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',1300,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random_1gb\\randValid\\CASIA_ValidationSet_Rand1GB_')
    validationPreprocessing.StartProcess(3,10)

def createTestSet():
    #testPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\testSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',2600,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\randTest\\CASIA_TestSet_Rand_')
    testPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\testSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',1300,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random_1gb\\randTest\\CASIA_TestSet_Rand1GB_')
    testPreprocessing.StartProcess(0,5)

def createTrainSet():
    #trainPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\trainSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',7800,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\randTrain\\CASIA_TrainSet_Rand_')
    trainPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\trainSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',3900,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random_1gb\\randTrain\\CASIA_TrainSet_Rand1GB_')
    trainPreprocessing.StartProcess(177,178)

print ("working In TrainingSet...")
createTrainSet()

print ("TrainSet OK")
print ("working In TestSet...")
createTestSet()
'''
print ("TestSet OK")
print ("working In ValidationSet...")
createValidationSet()
print ("ValidationSet OK")
'''
print "FINISH :)"