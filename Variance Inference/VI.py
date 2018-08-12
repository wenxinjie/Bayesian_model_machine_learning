from __future__ import division
import math
import pandas as pd
import numpy as np
import scipy.special
import matplotlib.pyplot as  plt
import pylab as py
import heapq

data_x=pd.read_csv('X_set3.csv')
data_y=pd.read_csv('y_set3.csv')
data_z=pd.read_csv('z_set3.csv')

# data_x.iloc[:1630,i]
a0 = 10**(-16)
b0 = 10**(-16)
e0 = 1
f0 = 1
d = 501
N=500
loop=500

log_gamma_a=np.math.lgamma(a0+1/2)
digamma_a=scipy.special.digamma(a0+1/2)
log_gamma_e=np.math.lgamma(e0+N/2)
digamma_e=scipy.special.digamma(e0+N/2)

#initial sigma and covirance
E_diag=np.eye(d)
E_lambda=1

#xi multiply xi and sum
sum_xi2=np.zeros((d,d))
for i in range(N):
	sum_i=data_x.iloc[i,:].reshape(1,-1)*data_x.iloc[i,:].reshape(-1,1)
	sum_xi2=sum_xi2+sum_i


#xi multiply yi and sum
sum_x_y=np.zeros((d,1))
for i in range(N):
	sum_i=data_x.iloc[i,:].reshape(-1,1)*data_y.iloc[i,0]
	sum_x_y=sum_x_y+sum_i

#produce new sigma and miu
def q_w(E_lambda,E_diag):
	sigma=np.linalg.inv(E_diag+E_lambda*sum_xi2)
	miu=np.dot(sigma,(E_lambda*sum_x_y))
	return sigma,miu

#calculate the expectation of (yi-xi*w)^2
def E_yi_xi_w(sigma,miu):
	sum_E=0
	for i in range(N):
		sum_i=(data_y.iloc[i,0]-np.dot(data_x.iloc[i,:],miu))**2+np.dot(np.dot(data_x.iloc[i,:],sigma),data_x.iloc[i,:])
		sum_E=sum_i+sum_E
	return sum_E

# produce new e and f
def q_lambda(sigma,miu):
	e=e0+N/2
	f=f0+E_yi_xi_w(sigma,miu)/2
	return e,f

#produce new a and b for alpha(i)
def q_alpha(i,sigma,miu):
	a=a0+1/2
	b=b0+(miu[i]**(2)+sigma[i][i])/2
	return a,b

#produce the new expectation of diag(alpha1,alpha2...)
def E_alpha(sigma,miu):
	E_diag=np.zeros((d,d))
	for i in range(d):
		a,b=q_alpha(i,sigma,miu)
		E_diag[i][i]=a/b
	return E_diag

#calculate the expextation of w*sigma*w
def E_w_sigma_w(sigma,miu):

	E_diag=E_alpha(sigma,miu)
	part=np.dot(miu.reshape(1,-1),np.dot(E_diag,miu.reshape(-1,1)))+np.trace(np.dot(E_diag,sigma))
	return part

#calculate the expectation of ln(lambda)
def E_q_ln_lambda(e,f):
	part=-np.math.log(f)+digamma_e
	return part

#calculate the expecation of ln(alpha)
def E_q_ln_alpha(a,b):
	part=-np.math.log(b)+digamma_a
	return part

#calculate the sum of expectation of alpha
def sum_E_ln_alpha(sigma,miu):
	sum_E_alpha=0
	for i in range(d):
		a,b=q_alpha(i,sigma,miu)
		temp=E_q_ln_alpha(a,b)
		sum_E_alpha=sum_E_alpha+temp
	return sum_E_alpha

#calculate the expectation of p(w) with q distribution
def E_q_ln_p_y(sigma,miu,e,f):

	part1=N*(E_q_ln_lambda(e,f))/2-(e/f)*E_yi_xi_w(sigma,miu)/2
	return part1

#calculate the expectation of p(λ) with q distribution
def E_q_ln_p_lambda(e,f):
	part1=(e0-1)*(E_q_ln_lambda(e,f))-f0*e/f
	return part1

#calculate the expectation of p(w) with the q distribution
def E_q_ln_p_w(sigma,miu):
	part1=sum_E_ln_alpha(sigma,miu)-E_w_sigma_w(sigma,miu)
	part2=part1/2
	return part2

#calculate the expectation of p(alpha) with the q distribution
def E_q_ln_p_alpha(sigma,miu):
	sum_e=0
	for i in range(d):
		a,b=q_alpha(i,sigma,miu)
		part=(a0-1)*(E_q_ln_alpha(a,b))-b0*a/b
		sum_e=sum_e+part
	return sum_e

#calculate the expectation of q(alpha) with the q distribution
def E_q_ln_q_alpha(sigma,miu):
	sum_e=0
	for i in range(d):
		a,b=q_alpha(i,sigma,miu)
		part=a*np.math.log(b)-log_gamma_a+(a-1)*(E_q_ln_alpha(a,b))-a
		sum_e=sum_e+part
	return sum_e

#calculate the expectation of q(λ) with the q distribution
def E_q_ln_q_lambda(e,f):
	part_1=e*np.math.log(f)-log_gamma_e+(e-1)*(E_q_ln_lambda(e,f))-e
	return part_1

#calculate the expexctation of q(w) witht the q distribution
def E_q_w(sigma,miu):
	log=np.linalg.slogdet(sigma)
	E_qw=-(d+log[0]*log[1])/2
	return E_qw

def L(e,f,sigma,miu):
	part1=E_q_ln_p_y(sigma,miu,e,f)+E_q_ln_p_lambda(e,f)+E_q_ln_p_w(sigma,miu)+E_q_ln_p_alpha(sigma,miu)
	part2=E_q_ln_q_alpha(sigma,miu)+E_q_ln_q_lambda(e,f)+E_q_w(sigma,miu)
	return part1-part2



sigma,miu=q_w(E_lambda,E_diag)
l_sum=np.zeros((loop))

Ealpha=np.zeros(d)
Ee_f=0

for i in range(loop):
	e,f=q_lambda(sigma,miu)
	if (i==(loop-1)):
		
		a=E_alpha(sigma,miu)
		for j in range(d):
			Ealpha[j]=1/a[j][j]
		Ee_f=f/e

	print(f/e)
	print(i)
	l=L(e,f,sigma,miu)
	sigma,miu=q_w(e/f,E_alpha(sigma,miu))
	l_sum[i]=l
print(l_sum[loop-1])
print(Ealpha)

fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.xlabel("k")
plt.ylabel("1/E_alpha_K")
plt.title("Dataset 3: Value of 1/E_alpha_K")
plt.plot(np.arange(0, d), Ealpha)
plt.show()


fig = plt.figure(2)
ax = fig.add_subplot(111)
plt.plot(np.arange(0, loop), l_sum[:])
plt.xlabel("itertaion")
plt.ylabel("L(v)")
plt.title("Dataset 3: Value of L(v)")
plt.show()


Y=np.zeros(N)
for i in range(N):
	Y[i]=np.dot(miu.T,data_x.iloc[i,:])
Z=np.zeros(N)
for i in range(N):
	Z[i]=10*np.sinc(data_z.iloc[i,0])
fig = plt.figure(3)
ax = fig.add_subplot(111)
plt.title('Dataset 3')
Y_, = ax.plot(data_z.iloc[:,0],Y,"r",label="xT*w",)


y_, = ax.plot(data_z.iloc[:,0],data_y.iloc[:,0],"b.",label="y")


z_, = ax.plot(data_z.iloc[:,0],Z,"k",label="10 ∗ sinc(z)")
plt.legend(loc='upper right')
plt.xlabel('Z')
plt.show()  





