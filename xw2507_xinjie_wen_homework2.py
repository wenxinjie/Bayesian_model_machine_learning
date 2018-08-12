from __future__ import division
import math
import pandas as pd
import numpy as np
import pylab as py
import matplotlib.pyplot as  plt
import heapq

c=0.1
deta=0.25
I=np.eye(10)
iteration=50
n_user=943
n_movie=1682
n_data=100000

R=pd.read_csv('ratings1.csv')

#initialize V


#calculate vj*vj and sum, rij*vj and sum

def U_V_r():
	u=[list() for i in xrange (n_user)]

	for row in range(0,n_data):
		u_i_r=[]
		u_i_r.append(R.iloc[row,1])
		u_i_r.append(R.iloc[row,2])
		u[R.iloc[row,0]-1].append(u_i_r)
	return u

def V_U_r():
	v=[list() for i in xrange (n_movie)]

	for row in range(0,n_data):
		v_i_r=[]
		v_i_r.append(R.iloc[row,0])
		v_i_r.append(R.iloc[row,2])
		v[R.iloc[row,1]-1].append(v_i_r)
	return v

def Vi_2(V,u):
	A=[]
	B=[]
	for i in range(0,n_user):
		sum2=np.zeros((10,10))
		sum1=np.zeros((10,1))

		for j in range(len(u[i])):
			v=V[u[i][j][0]-1]
			r=u[i][j][1]
			sum1+=v.reshape(-1,1)*r
			sum2+=v.reshape(-1,1)*v.reshape(-1,1).T
		B.append(sum1)
		A.append(sum2)
	return A,B

#calculate miu and sigma
def miu_sigma(A,B):
	sigma=[]
	miu=[]
	for i in range(0,n_user):
		temp_sigma=np.linalg.inv(I/c+A[i]/deta)

		temp=B[i]/deta

		temp_miu=np.dot(temp_sigma,temp.reshape(-1,1))

		miu.append(temp_miu)
		sigma.append(temp_sigma)
	return miu,sigma 

#calculate miu*miu.T*sigma and sum, rij*miu and sum
def miu_2(miu,sigma,v):
	C=[]
	D=[]
	for j in range(n_movie):
		sum_1=np.zeros((10,10))
		sum_2=np.zeros((10,1))

		for i in range(len(v[j])):
			sum_1 += np.dot(miu[v[j][i][0]-1],miu[v[j][i][0]-1].T)+sigma[v[j][i][0]-1]
			sum_2 += v[j][i][1]*miu[v[j][i][0]-1]
		C.append(sum_1)
		D.append(sum_2)
	return C,D

#calculate new V
def new_V(C,D):
	V_new=[]
	for j in range(n_movie):
		temp_1=np.linalg.inv(I/c+C[j]/deta)
		temp_2=D[j]/deta
		V_new.append(np.dot(temp_1,temp_2.reshape(-1,1)))
	return V_new



#calculate L
def L(C,D,V_new):
	L=[]
	for j in range(n_movie):
		tempt_1=(-0.5/deta)*((np.dot(np.dot(V_new[j].reshape(1,-1),C[j]),V_new[j]))-2*np.dot(D[j].reshape(1,-1),V_new[j]))
		tempt_2 = np.dot(V_new[j].reshape(1,-1),V_new[j])/(2*c)
		L.append(tempt_1-tempt_2)
	sum_L=0
	for j in range(n_movie):
		sum_L+=L[j]
	return sum_L

def Var(u_m,v_m,V,sigma):
	var=[list() for i in xrange(len(u_m))]
	for i in range(len(u_m)):
		tem=np.zeros(len(v_m))
		for j in range(len(v_m)):
			var_temp=deta+np.dot(np.dot(V[v_m[j]].reshape(1,-1),sigma[u_m[i]]),V[v_m[j]])
			tem[j]=var_temp

		var[i]=tem
	return var

def main():
	V=np.random.normal(size=(n_movie,10))
	L_sum=np.zeros([iteration])
    
	u=U_V_r()
	v=V_U_r()

	for i in range(iteration):
	
		A,B=Vi_2(V,u)
		miu,sigma=miu_sigma(A,B)
		C,D=miu_2(miu,sigma,v)
		V=new_V(C,D)
		L_sum[i]=L(C,D,V)

	print(L_sum)
	plt.figure(2)
	plt.plot(np.arange(0, iteration), L_sum)
	plt.xlim(2,55)
	plt.xlabel("itertaion")
	plt.ylabel("L(v)")
	plt.title("Value of L(v)")
	plt.show()

	u_m=[18, 33, 35, 92, 142, 146, 165, 201, 241, 299, 308, 363]
	v_m=[49, 99, 180, 257, 173, 126, 285, 0, 97, 287, 55, 299, 171, 293, 6, 312, 120, 236, 116, 78]
	var=Var(u_m,v_m,V,sigma)

	print(var[0])

	for i in range(len(u_m)):
		plt.subplot(3,4,i+1)
		plt.title("user: "+str(u_m[i]))
		for j in range(len(v_m)):
			plt.vlines(v_m[j], 0, var[i][j],colors = "c")

	plt.show()




main()
u=U_V_r()
v=V_U_r()
sum_set=np.zeros(n_movie)
for j in range(n_movie):
	sum=0
	for i in range(len(v[j])):
		sum+=v[j][i][1]
	sum_set[j]=sum
print(sum_set)
print("\n")
print(np.shape(sum_set))
print("movies:")
print(heapq.nlargest(20, range(len(sum_set)), sum_set.take))
print("\n")

rating=np.zeros(n_user)
for i in range(n_user):
	times=len(u[i])
	rating[i]=times
print(rating)
print("ratings")
print(heapq.nsmallest(12, range(len(rating)), rating.take))










	






