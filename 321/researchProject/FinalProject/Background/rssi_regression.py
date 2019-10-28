#!/usr/bin/env python
# coding: utf-8
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Author: Xavier Theo Quinn
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
I am going to utilize regression to go from rssi to a distance.
'''
#Imports
import csv
import matplotlib.pyplot as plt
import math
import copy
import random
import statistics as stat
import ast

from string import ascii_lowercase
from datetime import datetime
from random import randint
import itertools
from statistics import mean
from scipy.spatial import distance

import numpy as np
from sklearn import linear_model

import ast



#SCIKIT-LEARN
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ##Methods:
# #General
#Format print
# VERBOSE=False
VERBOSE=True

def fprint(label, value) :
	try: VERBOSE
	except NameError: print("{0}: {1}\n".format(label, value))
	else:
		if VERBOSE :
			print("{0}: {1}\n".format(label, value))


#Quick print
def qprint(*args) :
	var=args[len(args)-1]
	for i in range(len(args)-1) :
		print(args[i], end ="")

	print("{0}: {1}".format(var, repr(eval(var)))) #yeah yeah, insecure methods, but I doubt anyone will be tryig to inject code into my ML hw.

#prints out lists of lists in the format most commonly found in this data.
def lprint(label,toprint) :
	# print(label)
	for data in toprint :
		# for variableName in data :
		fprint(label,data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#cleans CSV
def column2Float(dataset,column) :
	dataset[column]=[float(data.strip()) for data in dataset[column]]

#loads Data
def load_data(filename) :
	with open(filename, 'r') as f:
		data=csv.reader(f)
		rowList=[]
		for row in data:
			colList=[]
			for column in row :
				try :
					# fprint("column",column)
					colList.append(ast.literal_eval(column))
				except :
					print("csv error")
			rowList.append(colList)
		return rowList

#imports and cleans specified CSV
def importAndCleanCSV(filename) :
	loadedData=load_data(filename)

	# Apply to the loaded Swedish auto data here
	for i in range(len(loadedData)) :
		column2Float(loadedData,i)

	print("{0}:\ninstances: {1} \nfeaturs: {2}".format(filename, len(loadedData), len(loadedData[0])))
	return loadedData


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#takes in full instances, including the class, and returns euclidean distance
def get_euc_dist(inst1, inst2) :
	# print("inst1 {0}, inst2 {1}".format(inst1, inst2))
	return distance.euclidean(inst1, inst2[:-1])

def rssi_to_dist(rssi, n):
	# rssi=int(re.sub("[^0-9\-]", "", rssi))
	txPow=-50 #Standard transmission power for bluetooth, could be updated to be dynamic.
	return math.pow(10, ((txPow-rssi)/(n*10)))

def dist_to_n(rssi, dist):
	rssi=int(re.sub("[^0-9\-]", "", rssi))
	txPow=-50 #Standard transmission power for bluetooth, could be updated to be dynamic.
	return math.pow(10, ((txPow-rssi)/(n*10)))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

filename="testOut.csv"
dataset=load_data(filename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
features=[]
log_features=[]
results=[]

# Loads in the specific format of the CSV

for entry in dataset :
	device_loc=entry[1]

	for beacon in entry[0] :

		if beacon[1]>-100 : #this is to remove the outliers shown in figure 5
			dist=get_euc_dist(beacon[0],device_loc)
			results.append(dist)
			# fprint("beacon[1]",beacon[1])
			log_features.append([math.log10(abs(beacon[1]))])
			features.append([beacon[1]])

lprint("coc",features)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Scikit Stuff
X_train, X_test, y_train, y_test = train_test_split(features, results, test_size=0.2, random_state=0)
rssi_model = LinearRegression()#, tol=1e-3)
rssi_model.fit(X_train, y_train)
fprint("score",rssi_model.score(X_test, y_test))
predicted=rssi_model.predict([[-152]])
fprint("predicted",predicted[0])
'''
This model supossedly converts rssi to distance

'''

# plot prediction and actual data
y_pred = rssi_model.predict(X_test)
# plt.plot(y_test, y_pred, 'b.')

# plot a line, a perfit predict would all fall on this line
# x = np.linspace(0,len(), 100)
# y = x
# plt.plot(x, y)

plt.subplot(1,2,1)
plt.plot(results,features, 'r.')
# plt.plot(np.unique(results), np.poly1d(np.polyfit(results,features, 1))(np.unique(results)))
plt.title('rssi vs dist with outliers removed, groups of 8 (v1)')
plt.xlabel("Predicted distance (relative)")
plt.ylabel("rssi (dbm)")

plt.subplot(1,2,2)
plt.plot(results,log_features, 'b.')

plt.title('rssi vs dist with outliers removed, log10 (v1)')
plt.xlabel("Predicted distance (relative)")
plt.ylabel("rssi (dbm)")

plt.show()

# predicted_dist_set=[]
# for entry in dataset :
# 	new_loc_list=[]
# 	for loc in entry[0] :
# 		loc=list(loc)
# 		# fprint("loc[1]",loc[0])
# 		predicted=rssi_model.predict([[loc[1]]])
# 		# fprint("predicted",predicted)
# 		new_loc_list.append([loc[0],predicted[0]])
# 	# fprint("loclist",new_loc_list)
# 	predicted_dist_set.append([new_loc_list,entry[1]])
#
#
# lprint("dataset",predicted_dist_set)
# # print(rssi_model.predict())
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



'''
https://scikit-learn.org/stable/tutorial/basic/tutorial.html#model-persistence
For regression of the rssi data I have about 4200 values, as each of my instances has at least 3 beacon values, and I only need one for rssi.
And according to the flow chart, as I have significantly less than 100K instances, I should utilize SGD, which I'm okay with.
However, due to the nature of converting RSSI signals to distance (attenuation and logarithmic), I am going to compare this to a polynomial SVR algorithm

I just realized that the rssi regression is single feature, which seems dumb to use ML for, but whatev, lets see how it turns out.

So I used regression to go from rssi to distance, and it sucked, which made me plot out my actual data, which is also not great.
This is concerning as there could be issues at almost any point in my code.


next I was going to approach this by solving n, then calculating dist.


Stochiastic gradient descent as minimization of overall distance function

'''
