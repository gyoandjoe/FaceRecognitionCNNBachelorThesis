__author__ = 'Giovanni'

import random

def MakeRowsRandom(sourceName,targetName):
    with open(sourceName,'r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(targetName,'w') as target:
        for _, line in data:
            target.write( line )

MakeRowsRandom("E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\CASIAFULL.csv","E:\\My Documents\\BUAP\\Titulacion\\Tesis\\Resources\\Data Sets\\CASIA Processing\\Results\\CASIAFULL_RANDOM.csv")