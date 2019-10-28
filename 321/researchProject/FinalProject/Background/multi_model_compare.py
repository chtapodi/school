#!/usr/bin/env python
# coding: utf-8

# # CSC-321: Data Mining and Machine Learning
#
# ## Working with scikit-learn
#
# In this notebook, I'll walk you through some elements of working with scikit learn.
# For documentation, refer to: https://scikit-learn.org/stable/index.html
#
# This notebook requires the Pima Indians diabetes data set.
#
# I've used two new libraries.
# (1) scikit learn (sklearn)
# (2) pandas
#
# Pandas isn't strictly necessary, because there are other methods you can use to load and slice data. But it's one of the most highly used python libraries for data manipulation.
#
# It basically brings the power of the data frame from R into python. You can find more here: https://pandas.pydata.org/
#
# In this notebook there will be a demonstration of:
# - preparing the data with pandas
# - putting it into the right format for scikit learn
#     - X values for training
#     - y values for testing
#     - in numpy array format
# - performing cross-validations with a variety of models
# - graphing the results
# - performing a t-test between selected models
# - then performing feature selection
# - slicing out irrelevant features
# - re-running one of the models on the new features

# In[24]:


# Compare Algorithms
# From scikit learn tutorial
# With modification by Nick Webb


# Pandas is an important data manipulation library
# You don't have to use it for your project, but I include it
# here so you know about it.



import pandas
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

import csv
import matplotlib.pyplot as plt
import math
import copy
import random
import statistics as stat
import ast


# Import models

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

# And import the feature selection mechanism

from sklearn.feature_selection import RFE
from sklearn.metrics.scorer import make_scorer






from string import ascii_lowercase
from datetime import datetime
from random import randint
import itertools
from statistics import mean
from scipy.spatial import distance

import numpy as np
from sklearn import linear_model

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


# #CSV functions

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
					colList.append(column)
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


def writeCSV(filename, csv_list) :
	#Write the data to a csv so I don't need to melt my computer every time I need this data
	with open(filename, 'w', newline='') as myfile:
		 wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		 wr.writerows(csv_list)


def column2Float(dataset,column) :
	dataset[column]=[float(data.strip()) for data in dataset[column]]




def get_grid_index(grid, val):
	for row in grid:
		if val in row:
			# fprint("grid", grid)
			x=grid.index(row)
			y=row.index(val)
			# print("val {0}, x {1}, y {2}".format(val, x, y))
			return [x, y]




def loc_to_pos(loc_list, grid_side_length) :
	grid=grid_gen(grid_side_length)
	to_return=[]
	for val in loc_list :
		indexes=get_grid_index(grid, val)
		# fprint("val",indexes)
		to_return.append(indexes)
	return to_return



def indev_euc_dist(inst1, inst2) :
	# print("inst1 {0}, inst2 {1}".format(inst1, inst2))
	return distance.euclidean(inst1, inst2)


#this generates the grid which will be used to assign values to locations. Values are assigned via grid_gen[x][y]
def grid_gen(side_length) :
	grid=[0]*side_length
	for row_index in range(side_length) :
		row=[0]*side_length
		for col_index in range(side_length) :
			row[col_index]=col_index*side_length+(row_index)
		grid[row_index]=row
	return grid

#custom scoring
def cust_scorer(x_val, y_val) :
	x_loc=loc_to_pos(x_val, 4)
	y_loc=loc_to_pos(y_val, 4)
	score=[]
	for i in range(len(x_val)) :
		# score.append(indev_euc_dist(x_loc[i], y_loc[i]))
		score.append((indev_euc_dist(x_loc[i], y_loc[i])+1)**2) #This is to increase the weight of one
	return sum(score)


my_scorer = make_scorer(cust_scorer, greater_is_better=False)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
filename="3groupmultiregression.csv"
loaded_data=load_data(filename)
my_data=[]
for instance in loaded_data :
	grid_list=ast.literal_eval(instance[0])
	for i in range(len(grid_list)) :
		if grid_list[i]==0 :
			grid_list[i]=-200
	location=instance[1]
	# fprint("instance",location)
	grid_list.append(int(location))
	my_data.append(grid_list)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





fprint("my_data[0]",my_data[0])
# my_data=importAndCleanCSV("4grid_class.csv")
# lprint("my_data",my_data)


dataframe=pandas.DataFrame.from_records(my_data)
# print(dataframe)
names=[]
for i in range(len(my_data[0])) :
	names.append("square {0}".format(i))

print(dataframe)

# get data from data frames, as numpy arrays
# note that by convention, we use X for input features
# and lower case y for the target class
print("my data")
array = dataframe.values
X = array[:,0:16]
y = array[:,16]
fprint("X", X)
fprint("y", y)


# prepare configuration for cross validation test harness
seed = 1

# prepare models
models = []
models.append(('ZR', DummyClassifier(strategy="most_frequent")))
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('KN5', KNeighborsClassifier()))
models.append(('KN7', KNeighborsClassifier(n_neighbors=7)))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('LIN', SVC(kernel='linear',gamma='auto')))
models.append(('RF',RandomForestClassifier(n_estimators=100)))
models.append(('NN',MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))

# evaluate each model in turn
# note that I'm going to run through each model above
# performing a 10-fold cross-validation each time
# (n_splits = 10), specifying 'accuracy' as my measure


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

results = []
classifiers = []
# scoring = 'accuracy'
scoring = my_scorer
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
	fprint("model",model)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	classifiers.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# fprint("FUCKING SWEET JESUS", models[2])

# boxplot algorithm comparison

fig = plt.figure()
fig.suptitle('Algorithm Comparison, custom euclidean scoring metric, full grid w/ 1/5th resolution')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(classifiers)
plt.show()

print('\n***Performing t-tests***\n\n')


ttest,pval = stats.ttest_rel(results[0], results[1])
print('P-Val between ZeroR and Logistic Regression: %.2f' % pval)

if pval<0.05:
	print("reject null hypothesis")
else:
	print("accept null hypothesis")

print()

ttest,pval = stats.ttest_rel(results[1], results[5])
print('P-Val between Logistic Regression and Naive Bayes: %.2f' % pval)

if pval<0.05:
	print("reject null hypothesis")
else:
	print("accept null hypothesis")

print('\n\n***Examining Features***\n\n')

log = LogisticRegression(solver='liblinear')
rfe = RFE(estimator=log, step=1)
rfe.fit(X, y)
print("Feature names:",names)
print("Feature ranking:",rfe.ranking_)

print('\n\n***Slicing data to include ONLY features ranked 1***\n\n')

# I use pandas (badly) to do this, slicing by column names
# I extract the column names from the ranking, above
# There's better ways to do this, but it's late and I'm tired
#
# newCols = []
# index = 0
# for i in rfe.ranking_:
# 	if i == 1:
# 		newCols.append(names[index])
# 	index+=1
# newCols.append('class')
# newData = dataframe[dataframe.columns[dataframe.columns.isin(newCols)]]
#
# fprint("newData",newData)
#
#
# # Extract the training and test data from the pandas data frame
#
# array = newData.values
#
# X = array[:,0:414]
# y = array[:,414]
#
# # I'm going to perform a single 10-fold cross-validation
# # Using my new data, and just two models
# # Naive Bayes and Logistic Regression
#
# kfold = model_selection.KFold(n_splits=10, random_state=seed)
#
# cv_results_1 = model_selection.cross_val_score(GaussianNB(), X, y, cv=kfold, scoring=scoring)
# msg = "%s: %f (%f)" % ('NB', cv_results_1.mean(), cv_results_1.std())
# print(msg)
#
# cv_results_2 = model_selection.cross_val_score(LogisticRegression(solver='liblinear'), X, y, cv=kfold, scoring=scoring)
# msg = "%s: %f (%f)" % ('LG', cv_results_2.mean(), cv_results_2.std())
# print(msg)



'''
So this gives some neat little insight into the fact that the squares near the beacons are the most important by far
Feature ranking: [208 207 206 205 204 203 202 201 200 199 198 197 196 195 194 193 192 191
 190 189 188 187 186 185 184 183 182 181 180 179 178 177 176 175 174 173
 172 171 170 169 168 167 166 165 164 163 162 161 160 159 158 157 156 155
 154 153 152 151 150 149 148 147 146 145 144 143 142 141 140 139 138 137
 136 135 134 133 132 131   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   2   4   6   8  10  12  14   1  18  20  22  24  26  28  30  32  34
  36  38  40  42  44  46  48  50  52  54  56  58  60  62  64  66  68  70
   1  74  76  78  80  82  84  86  88  90  92  94  96  98 100 102 104 106
 108 110 112 114 116 118 120 122 124 126 128 130 129 127 125 123 121 119
 117 115 113 111 109 107 105 103 101  99  97  95  93  91  89  87  85  83
  81  79  77  75  73   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   3
   5   7   9  11  13  15  16  17  19  21  23  25  27  29  31  33  35  37
  39  41  43  45  47  49  51  53  55  57  59  61  63  65  67  69  71  72]
'''
