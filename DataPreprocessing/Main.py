from DataPreprocessing.PreprocessingCASIA import PreprocessingCASIA

__author__ = 'Giovanni'

#objPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\trainSet.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',7800,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\Train\\CASIA_TrainSet_NoRandom_')




def createValidationSet():
    validationPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\validationSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',2600,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\CASIA_ValidationSet_Rand_')
    validationPreprocessing.StartProcess(2,6)

def createTestSet():
    testPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\testSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',2600,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\CASIA_TestSet_Rand_')
    testPreprocessing.StartProcess(2,6)

def createTrainSet():
    trainPreprocessing = PreprocessingCASIA("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\trainSet_rand.csv",'E:\\My Documents\BUAP\\Titulacion\\Tesis\\CASIA\\DataBase\\Normalized_Faces\\webface\\100\\',7800,'E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\Distribute and random\\CASIA_TrainSet_Rand_')
    trainPreprocessing.StartProcess(5,6)

print ("working...")
#createTrainSet()
print ("TrainSet OK")
#createTestSet()
print ("TestSet OK")
createValidationSet()
print ("ValidationSet OK")
print "FINISH :)"