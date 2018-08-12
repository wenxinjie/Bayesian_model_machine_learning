from __future__ import division
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import pylab as py
import csv
import random
import copy
import time
import heapq


a0=4.5
b0=0.25
alpha=0.75
loops=50

def read_file():
	b=[]
	with open('x copy.csv') as f:
		f_csv = csv.reader(f)
		for row in f_csv:
			b.append(int(row[0]))
	a=[[],[]]
	d=np.arange(2000)
	a[0]=np.array(b)
	for i in range(2000):
		d[i]=random.randint(1,30)
	a[1]=d

	return a

def random_pick(some_list, probabilities):  
      x = random.uniform(0, 1)  
      cumulative_probability = 0.0  
      for item, item_probability in zip(some_list, probabilities):  
            cumulative_probability += item_probability  
            if x < cumulative_probability: break  
      return item 


def initial_cluster(data):
	a=max(data[1])
	cluster_list=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	sum_a1=[]
	for i in range(2000):
		ip=data[1][i]
		cluster_list[ip-1].append(data[0][i])
	for j in range(len(cluster_list)):
		sum_a1.append(sum(cluster_list[j]))
	return cluster_list,sum_a1

def count_in_cluster(cluster):
	count_list=[]
	for i in range(len(cluster)):
		count_list.append(len(cluster[i]))

	return count_list

def initial_lamda(sum_a,count_n):
	lamda_list=[]
	for j in range(len(sum_a)):
		a=sum_a[j]+a0
		b=count_n[j]+b0
		lamda_list.append(random.gammavariate(a, 1/b))
	return lamda_list

def sample_lamda_ci(value):
	a=a0+value
	b=b0+1
	c=random.gammavariate(a, 1/b)
	return c


def sample_ci(i,data,cluster,sum_a,lamda,count_n):
	index=data[1][i]
	fai=[]

	for j in range(len(count_n)):
		if j==index-1:
			if count_n[index-1]==1:
				continue
			else:
				pro=(count_n[index-1]-1)*math.exp(-lamda[j]+data[0][i]*math.log(lamda[j]))
				fai.append(pro)
		else:
			pro=(count_n[j]*math.exp(-lamda[j]+data[0][i]*math.log(lamda[j])))
			fai.append(pro)

	pro_1=alpha*math.exp(a0*math.log(b0)+np.math.lgamma(data[0][i]+a0)-(data[0][i]+a0)*math.log(b0+1)-np.math.lgamma(a0))
	fai.append(pro_1)
	sum_fai_1=sum(fai)
	for k in range(len(fai)):
		fai[k]=fai[k]/sum_fai_1
	tem_ci=random_pick(np.arange(1,len(fai)+1),fai)


	if tem_ci==len(fai):
		# print(">>>>>>>>>     new !  new !    <<<<<<<<<")
		new_lamda=sample_lamda_ci(data[0][i])
		lamda.append(new_lamda)
		data[1][i]=tem_ci
		sum_a.append(data[0][i])
		count_n.append(1)
		cluster.append([data[0][i]])
	else:
		sum_a[tem_ci-1]+=data[0][i]
		count_n[tem_ci-1]+=1
		cluster[tem_ci-1].append(data[0][i])
		data[1][i]=tem_ci
	sum_a[index-1]-=data[0][i]
	count_n[index-1]-=1
	cluster[index-1].remove(data[0][i])

	if count_n[index-1]==0:
		del sum_a[index-1]
		del lamda[index-1]
		del count_n[index-1]
		del cluster[index-1]
		data[1][data[1]>index]=data[1][data[1]>index]-1
	return sum_a,count_n,cluster


def sample_lamda(count_n,sum_a,cluster):
	lamda_list=[]
	for j in range(len(count_n)):
		a=sum_a[j]+a0
		b=count_n[j]+b0
		lamda_list.append(random.gammavariate(a, 1/b))
	return lamda_list

data=read_file()
cluster,sum_a=initial_cluster(data)
count_n=count_in_cluster(cluster)
lamda=initial_lamda(sum_a,count_n)
print(cluster)
print(sum_a)
print(count_n)
print(lamda)

sort=np.zeros((6,loops))
len_count=np.zeros(loops)

for loop in range(loops):
	print(">>>>>>>>>>>>>>>>>>>>   loop      >>>>>>>>>>  :  "+str(loop))
	for i in range(2000):
		sum_a,count_n,cluster=sample_ci(i,data,cluster,sum_a,lamda,count_n)
	lamda=	sample_lamda(count_n,sum_a,cluster)
	len_count[loop]=len(count_n)
	copy_count=copy.copy(count_n)
	a=heapq.nlargest(6, range(len(np.array(copy_count))), np.array(copy_count).take)
	for q in range(6):
		if q>=len(count_n):
			sort[q][loop]=0
		else:
			sort[q][loop]=count_n[a[q]]


fig = plt.figure(2)

ax = fig.add_subplot(111)
x1_, = ax.plot(np.arange(loops),sort[0],"b",label="y1")
x2_, = ax.plot(np.arange(loops),sort[1],"r",label="y2")
x3_, = ax.plot(np.arange(loops),sort[2],"g",label="y3")
x4_, = ax.plot(np.arange(loops),sort[3],"c",label="y4")
x5_, = ax.plot(np.arange(loops),sort[4],"m",label="y5")
x6_, = ax.plot(np.arange(loops),sort[5],"y",label="y6")

plt.xlabel("TIMES")
plt.ylabel("COUNT_NUMBER")
plt.show()

fig = plt.figure(3)
ax = fig.add_subplot(111)
y_, = ax.plot(data[0],data[1],"b.",label="y")
plt.xlabel("DATA")
plt.ylabel("Cluster")
plt.show()

fig = plt.figure(1)
ax = fig.add_subplot(111)
y_, = ax.plot(data[0],data[1],"b.",label="y")
plt.xlabel("DATA")
plt.ylabel("Cluster")
plt.show()








