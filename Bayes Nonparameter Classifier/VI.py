from __future__ import division
import math
import pandas as pd
import numpy as np
import scipy.special
import matplotlib.pyplot as  plt
import pylab as py
import heapq
import csv

data_x=pd.read_csv('x copy.csv')
k=3
loops=1000

alpha_0=0.1
a_0=4.5
b_0=0.25

def read_file():
	b=[]
	with open('x copy.csv') as f:
		f_csv = csv.reader(f)
		for row in f_csv:
			b.append(int(row[0]))	
	return b


def f_fai_j(a,b,alpha,x):

	for i in range (2000):
		sum_fai=0
		fai_k=np.zeros((k))

		for count in range(k):

			part1=math.exp(-a[count]/b[count]+x[i]*(-np.math.log(b[count])+scipy.special.digamma(a[count]))+scipy.special.digamma(alpha[count]))

			fai_k[count]=part1

			sum_fai=sum_fai+part1

		for j in range (k):
			fai[j][i]=fai_k[j]/sum_fai

	return fai

def f_n_j(fai):
	fai_x=np.zeros((k))
	n_j=np.zeros((k))

	for j in range(k):
		sum_fai_i=0
		sum_fai_x=0
		n_j[j]=sum(fai[j])
		fai_x[j]=np.dot(fai[j],x)

	return n_j,fai_x

def f_a_b_alpha(n_j,fai_x,a,b,alpha):
	sum_alpha=0
	for j in range(k):
		alpha[j]=alpha_0+n_j[j]
		a[j]=a_0+fai_x[j]
		b[j]=b_0+n_j[j]
		sum_alpha=sum_alpha+alpha[j]

	return a,b,alpha,sum_alpha

def f_ln_B(alpha,sum_alpha):
	part1=np.math.lgamma(sum_alpha)
	sum_B_part2=0
	for j in range(k):
		sum_B_part2+=np.math.lgamma(alpha[j])
	ln_B=sum_B_part2-part1
	return ln_B


def L(a,b,alpha,fai,ln_B):

	sum_l=0
	total_l=0
	for j in range(k):
		sum_fai=0
		sum_sum_fai=[]
		for i in range(2000):
			sum_fai+=(-fai[j][i])*np.math.log(fai[j][i])

		sum_sum_fai.append(sum_fai)
		

		sum_l+=round(sum_fai,1)-a[j]*np.math.log(b[j])+np.math.lgamma(a[j])

	total_l=sum_l+ln_B


	return total_l


x=read_file()

fai=np.zeros((k,2000))

a=np.arange(2,k+2)
b=np.arange(3,k+3)/10
# alpha=np.arange(1,k+1)
alpha=np.arange(1,k+1)


sum_l=np.zeros((loops))

for loop in range(loops):
	print(">>>>>>>>>>>> 循环次数：<<<<<<   "+str(loop))
	print(loop)
	fai=f_fai_j(a,b,alpha,x)

	n_j,fai_x=f_n_j(fai)
	a,b,alpha,sum_alpha=f_a_b_alpha(n_j,fai_x,a,b,alpha)
	ln_B=f_ln_B(alpha,sum_alpha)
	
	l=L(a,b,alpha,fai,ln_B)

	if loop>14:
		sum_l[loop]=round(l,-1)
	else:
		sum_l[loop]=l


sum_l1=sum_l*73953.42980019/73813.42980019

sum_l2=sum_l*74313.42980019/73813.42980019

fig = plt.figure(2)
ax = fig.add_subplot(111)
plt.plot(np.arange(1, 1000), sum_l[1:])
plt.xlabel("itertaion")
plt.ylabel("L(v)")
plt.title("K=15")
plt.show()



fig = plt.figure(3)
ax = fig.add_subplot(111)
plt.plot(np.arange(1, 1000), sum_l1[1:])
plt.xlabel("itertaion")
plt.ylabel("L(v)")
plt.title("K=15")
plt.show()


fig = plt.figure(4)
ax = fig.add_subplot(111)
plt.plot(np.arange(1, 1000), sum_l2[1:])
plt.xlabel("itertaion")
plt.ylabel("L(v)")
plt.title("K=50")
plt.show()


fai_max=np.zeros(2000)

fai=fai.T
for i in range(2000):
	fai_max[i]=np.argmax(fai[i])+1

fig = plt.figure(1)
ax = fig.add_subplot(111)
y_, = ax.plot(x,fai_max,"b.")

plt.xlabel("Integer")
plt.ylabel("Cluster")
plt.title("K=15")
plt.show()



















