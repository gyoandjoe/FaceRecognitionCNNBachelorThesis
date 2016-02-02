__author__ = 'Giovanni'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'E:\dev\TesisTest\logModel3\trainResults.csv', names=["epoch","cost"],usecols=[0,1],)
df = df.loc[df['epoch'] != 3]

results =  df.groupby('epoch', as_index=False)['cost'].mean()

xAll = df['epoch']
yAll = df['cost']
x = results['epoch']
y = results['cost']
plt.plot(x,y)
#plt.plot(xAll,yAll,'*')
plt.show()
print "OK"