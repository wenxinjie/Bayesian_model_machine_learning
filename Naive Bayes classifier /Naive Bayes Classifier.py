from __future__ import division
import math
import pandas as pd
import numpy as np

#import data of train and test
train_x=pd.read_csv('X_train.csv')
test_x=pd.read_csv('X_test.csv')

#sum of x when true(spam) or false(nonspam)
xtrue_sum=[]
xfalse_sum=[]
dif=[]

#set to store the probability of email as spam or nonspam
trueset=[]
falseset=[]

#counter of spam(t) and nonspam(f)
t=0
f=0

#sum x, numbers of spam and nonspam in training specimen
for i in range(54):
	column_sum = train_x.iloc[:1630,i].sum()
	xtrue_sum.append(column_sum)

for i in range(54):
	column_sum = train_x.iloc[1631:,i].sum()
	xfalse_sum.append(column_sum)

#the probability concerned about y
ytrue=float(1632)/4142
yfalse=float(2510)/4142

#factorial function
def sum_log(num1,num2):
	temp=0
	for i in range(num1,num2+1):
		temp=temp+math.log(i)
	return temp


#computer the probability of a email x1 as spam and nonspam
for i in range(461):
	pro_true=0
	pro_false=0
	for d in range(0,54):
		xd=test_x.iloc[i,d]
#culcating the probability of email as spam in test specimen
		p_true=(xtrue_sum[d]+1)*math.log(1632)+sum_log((xtrue_sum[d]+1),(xtrue_sum[d]+xd))-(xtrue_sum[d]+1+xd)*math.log(1633)-sum_log(2,(xd))+math.log(ytrue)
		pro_true=p_true+pro_true

#culcating the probability of email as spam in test specimen
		p_false=(xfalse_sum[d]+1)*math.log(2510)+sum_log((xfalse_sum[d]+1),(xfalse_sum[d]+xd))-(xfalse_sum[d]+1+xd)*math.log(2511)-sum_log(2,(xd))+math.log(yfalse)
		pro_false=p_false+pro_false

#store all the probabilities 	
	trueset.append(pro_true)
	falseset.append(pro_false)

print (trueset) 
print(falseset) 


#judge spam or nonspam according to the probability respectively
for i in range(461):
	if trueset[i]>falseset[i]:
		print(str(i)+"th email is spam")
		t+=1
	else:
		print(str(i)+'th email is non spam')
		f+=1

#the differences of two probability
for i in range(461):
	temp=trueset[i]-falseset[i]
	dif.append(temp)
print(dif)
print(t)
print(f)

for index,item in enumerate(dif):
	print index,item

#sort the difference
print(sorted(dif))


