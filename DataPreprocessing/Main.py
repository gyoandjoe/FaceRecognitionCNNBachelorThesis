from DataPreprocessing.PreprocessingCASIA import PreprocessingCASIA

__author__ = 'Giovanni'

#objPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\trainSet.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',7800,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\Train\\CASIA_TrainSet_NoRandom_')



print ("OK")
def createValidationSet():
    validationPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\validationSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',2600,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\CASIA_ValidationSet_Rand_')
    validationPreprocessing.StartProcess(0,2)

def createTestSet():
    testPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\testSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',2600,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\CASIA_TestSet_Rand_')
    testPreprocessing.StartProcess(0,2)

def createTrainSet():
    trainPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\trainSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',7800,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\CASIA_TrainSet_Rand_')
    trainPreprocessing.StartProcess(10,20)


createTestSet()
createValidationSet()
print "FINISH :)"