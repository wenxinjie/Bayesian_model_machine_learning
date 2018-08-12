from __future__ import division
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import pylab as py


data_x=pd.read_csv('x copy.csv')

k=9
loop=50

def fai_j(lam,pie,fai):
	for i in range(2000):
		fai_k=np.zeros((k))
		sum_fai=0
		for count in range(k):
			part=math.exp(-lam[count]+(data_x.iloc[i,0])*math.log(lam[count]))
			part_2=pie[count]
			fai_k[count]=part*part_2
			sum_fai=sum_fai+fai_k[count]
		for j in range(k):
			fai[i][j]= fai_k[j]/sum_fai
	return fai 

def lam_pie(fai):
	for j in range(k):
		sum_x=0
		sum_fai=0
		for i in range(2000):
			part1=fai[i][j]*data_x.iloc[i,0]
			part2=fai[i][j]
			sum_x=part1+sum_x
			sum_fai=part2+sum_fai
		pie[j]=sum_fai/2000
		lam[j]=sum_x/sum_fai
	return lam,pie

def L(lam,pie,fai):
	sum_l=0
	for i in range(2000):
		sum_ln=0
		for j in range(k):
			sum_ln+=pie[j]*(math.exp(-lam[j]+data_x.iloc[i,0]*math.log(lam[j])))
		sum_l+=math.log(sum_ln)

	return sum_l	

fai=np.zeros((2000,k))

lam=np.array([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5])
pie=np.array([0.066,0.0663,0.0665,0.0666,0.0668,0.0669,0.067,0.0667,0.0666,0.0666,0.0666,0.0666,0.666,0.0666,0.0676])

l=np.zeros((loop))
L_value=0
fai_max=np.zeros(2000)
for i in range(loop):
	fai=fai_j(lam,pie,fai)
	lam,pie=lam_pie(fai)
	L_value=L(lam,pie,fai)
	l[i]=L_value
	print(i)
	print(L_value)
print(l)

for i in range(2000):
	fai_max[i]=np.argmax(fai[i])+1

fig = plt.figure(1)
ax = fig.add_subplot(111)
y_, = ax.plot(data_x.iloc[:,0],fai_max,"b.")

plt.xlabel("DATA")
plt.ylabel("CLUSTER")
plt.title("K=15")
plt.show()

fig = plt.figure(2)
ax = fig.add_subplot(111)
plt.plot(np.arange(1, loop), l[1:])
plt.xlabel("ITERATION_TIMES")
plt.ylabel("f")
plt.title("K=3: Value of f")
plt.show()
