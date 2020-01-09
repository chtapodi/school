'''
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6308584/
https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html
'''
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
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#nicks stuff

import pandas
import matplotlib.pyplot as plt
from scipy import stats

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


def writeCSV(filename, csv_list) :
	#Write the data to a csv so I don't need to melt my computer every time I need this data
	with open(filename, 'w', newline='') as myfile:
		 wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		 wr.writerows(csv_list)
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


filename="3group.csv"
dataset=load_data(filename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
features=[]
log_features=[]
results=[]

# Loads in the specific format of the CSV

# for entry in dataset :
# 	device_loc=entry[1]
#
# 	for beacon in entry[0] :
#
# 		if beacon[1]>-100 : #this is to remove the outliers shown in figure 5
# 			dist=get_euc_dist(beacon[0],device_loc)
# 			results.append(dist)
# 			# fprint("beacon[1]",beacon[1])
# 			log_features.append([math.log10(abs(beacon[1]))])
# 			features.append([beacon[1]])

# lprint("coc",dataset)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This gets the center of the beacons for the instance
def get_center(beacon_list) :
	locations, rssi= zip(*beacon_list)
	min_x=locations[0][0]
	max_x=locations[0][0]
	min_y=locations[0][1]
	max_y=locations[0][1]
	for beacon in locations :
		if min_x>beacon[0] :
			min_x=beacon[0]
		if max_x<beacon[0] :
			max_x=beacon[0]
		if min_y>beacon[1] :
			min_y=beacon[1]
		if max_y<beacon[1] :
			max_y=beacon[1]
	return [(max_x-min_x)/2 + min_x, (max_y-min_y)/2 + min_y]

#conversts a list of locations to centered ones
def list_to_loc(values, center_list) :
	for i in range(len(values)) :
		values[i]=values[i]-center_list[i]

#centers an instance
def center_inst(instance) :
	center=get_center(instance[0])

	#center beacon vals
	for beacon in instance[0] :
		list_to_loc(beacon[0], center)

	#center relative rssi, even though this will be changed again
	list_to_loc(instance[1], center)

# #This converts all
# for instance in dataset :
# 	center_inst(instance)

# lprint("dataset",dataset)
'''All of the instances have been zero'd to their center '''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This section of code will "quadtrantize" the output data
#takes the zero'd location values and sorts them intro quadrants
def quatrandify(location) :
	quad=0
	if location[0]>0 :     #if x is on the right
		if location[1]>0 : #if y is on the top
			quad=1
		else :			   #if y is on the bottom
			quad=3
	else :				   #if x is on the left
		if location[1]>0 : #if y is on the top
			quad=0
		else :			   #if y is on the bottom
			quad=2
	return quad

# fprint("quadtrantized loc", dataset[0][1])
# fprint("quadtrantized loc", quatrandify(dataset[0][1]))
#
# for instance in dataset :
# 	instance[1]=quatrandify(instance[1])

# lprint("dataset",dataset)

'''
All of the instances have quadrants as outputs now

HOWEVER: I realized everything above this is practically useless
'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This the proper way to handle the above.

features, results= zip(*dataset)

# lprint("feat",features)

#this generates the grid which will be used to assign values to locations. Values are assigned via grid_gen[x][y]
def grid_gen(side_length) :
	grid=[0]*side_length
	for row_index in range(side_length) :
		row=[0]*side_length
		for col_index in range(side_length) :
			row[col_index]=col_index*side_length+(row_index)
		grid[row_index]=row
	return grid
# fprint("grid", grid_gen(3))s


#gets the local minimum for x and y, to be used to prep for gridifying
def get_min_max(location_list) :
	locations, rssi= zip(*location_list)
	x,y=zip(*locations)
	return [[min(x), max(x)],[min(y),max(y)]]

def scale(min,max, val) :
	scaled_value=(val-min)/(max-min)
	# fprint("min",min)
	# fprint("max",max)
	# fprint("val",val)
	# fprint("out", scaled_value)
	return scaled_value

def rssi_to_dist(rssi, n):
	# rssi=int(re.sub("[^0-9\-]", "", rssi))
	txPow=-50 #Standard transmission power for bluetooth, could be updated to be dynamic.
	return math.pow(10, ((txPow-rssi)/(n*10)))

def standardize_rssi(min_rssi, max_rssi, rssi) :
	n=2.9#see https://www.slideshare.net/slideshow/embed_code/key/cJTaUJ7qXumc8V for chosen value, but it really shouldnt matter.
	min_val=rssi_to_dist(max_rssi, 1.6) #So here is where things get a little weird. I am going to use variable n values to account for in room attenuation.
	max_val=rssi_to_dist(min_rssi, 6)

	# fprint("min_rssi",min_rssi)
	# fprint("max_rssi",max_rssi)
	# min_val=min_rssi
	# max_val=max_rssi

	# print("min {0} max {1}".format(min_val,max_val))
	dist=rssi_to_dist(rssi,n)
	scaled_dist=scale(min_val,max_val,dist)
	# fprint("dist",dist)
	# fprint("scaled_dist",scaled_dist)
	return scaled_dist

#This, given a location and the corosponding data, returns grid location
def get_grid_loc(location, min_max_list,grid_side_length) :
	grid=grid_gen(grid_side_length)
	coords=location

	for i in range(len(coords)) :
		# print("({0})-({1})={2}".format(coords[i],min_list[i],coords[i]-min_list[i))
		min_val=min_max_list[i][0]
		max_val=min_max_list[i][1]

		if min_val!=max_val : #avoiding dividing by zero
			val=int(scale(min_val,max_val, coords[i])*grid_side_length)
		else : #the case where there is one row
			val=0

		if val>grid_side_length :
			coords[i]=grid_side_length-1
		elif val>=1 :	#because list indexing starts at one less, but -1 is not an index
			coords[i]=val-1
		else :
			coords[i]=0
	grid_loc=grid[int(coords[0])][int(coords[1])]
	return grid_loc

# fprint("loc",get_grid_loc([19, 4],get_local_min(dataset[0][0]),3))



#This converts whole instances to grid referenced
#So each location in a list equates to a square
def inst_to_grid(instance, grid_side_length,min_rssi, max_rssi) :
	beacon_locations=instance[0]
	actual_location=instance[1]
	locational_feature_list=[0]*grid_side_length**2
	local_min=get_min_max(beacon_locations)

	# fprint("local_min",beacon_locations)
	loc, rssi= zip(*beacon_locations)

	for i in range(len(loc)) :
		# print("RSSI")
		locational_feature_list[get_grid_loc(loc[i],local_min,grid_side_length)]=standardize_rssi(min_rssi, max_rssi, rssi[i])
	actual_location=get_grid_loc(actual_location,local_min,grid_side_length)
	return[locational_feature_list,actual_location]



GRIDSIZE=4
location_list, out=zip(*dataset)

#Here I am getting the minimal and maximal rssi values across the entire dataset, which I will use to have a homogeniously scaled rssi output.
#HOWEVER There are some issues relating to outliers, which is causing my scaling to be way off.
flat_list=[item for group in location_list for item in group]
index, rssi = zip(*flat_list)
# lprint("location_list[0]",rssi)


min_rssi=min(rssi)
max_rssi=max(rssi)
# fprint()


grid_data_set=[]
for intance in dataset :
	grid_data_set.append(inst_to_grid(intance,GRIDSIZE, min_rssi, max_rssi))


writeCSV("4grid_class.csv", grid_data_set)
# fprint("mmmm", len(grid_data_set))
# lprint("gridified data", grid_data_set)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''Its time to start machine learning'''
features, results=zip(*grid_data_set)

lprint("data", grid_data_set)


X_train, X_test, y_train, y_test = train_test_split(features, results, test_size=0.2, random_state=0)


# scaler = StandardScaler()
# scaler.fit(X_train)  # Don't cheat - fit only on training data
# X_train = scaler.transform(X_train)
# y_train = scaler.transform(y_train)
# y_test = scaler.transform(y_test)
# X_test = scaler.transform(X_test)  # apply same transformation to test data

#train


# clf = DecisionTreeClassifier(criterion='entropy')

# clf = SGDClassifier(loss="log",penalty="elasticnet",tol = 0.00001,max_iter = np.ceil(10**6),verbose=1)#loss= 'log', alpha = 1, tol = 0.00001, max_iter = 1000, shuffle = False, random_state = 0
# clf = svm.LinearSVC(max_iter=np.ceil(10**6),verbose=1)


#HERE MAYBE

# clf=svm.SVC(degree=16, gamma='scale', kernel='poly', tol=.001,verbose=1)
# print(clf)
# clf.fit(X_train, y_train)
# print(clf.score(X_test,y_test))
# import graphviz
# from sklearn import tree

#print("Done")


# clf = clf.fit(X_train, y_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Nicks Stuff


names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv('pima-indians-diabetes.csv', names=names)


# get data from data frames, as numpy arrays
# note that by convention, we use X for input features
# and lower case y for the target class

array = dataframe.values
X = array[:,0:8]
y = array[:,8]


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

results = []
classifiers = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	classifiers.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)



# boxplot algorithm comparison

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
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



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #Tree print shit
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
#
# dot_data = tree.export_graphviz(clf, out_file=None,
# 					 feature_names=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"],
# 					 class_names=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"],
# 					 filled=True, rounded=True,
# 					 special_characters=True)
# graph = graphviz.Source(dot_data)
# graph
#
# fprint("score",clf.score(X_test, y_test))
# # predicted=rssi_model.predict([[-152]])
# # fprint("predicted",predicted[0])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#graphing Stuff
# y_pred = clf.predict(X_test)




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



pred_list=loc_to_pos(y_pred,GRIDSIZE)
act_list=loc_to_pos(y_test,GRIDSIZE)
# lprint("ypred",act_list)

'''
need to qrite graphing function, plots predicted and actual values on a grid and draws a line between them
'''
# fprint("pred_list[0]",pred_list[0])
# fprint("act_list[0]",act_list[0])





#This prints out lines between the real and predicted values
# for i in range(0, len(act_list), 2):
	# plt.plot(*zip(pred_list[i],act_list[i]), 'o-')




#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# plot prediction and actual data

plt.plot(y_pred, y_test, 'b.')
# plt.plot(X_test, y_test, 'b.')
# plt.plot(X_test, y_pred, 'r.')
plt.show()


'''
I realized that the method I implemented in previous research will not work with machine learning, as it would be really dumb.

So I beleive my current plan is to arrange and feed in the relative beacon locations and the rssi value as features, and have the output be a classifier of regions.
If I am going this route, it may make sense to degrade the beacon locations in the same manor.

I have decided to set the reference point as the average location for all beacons, and then divide this into four quadrants

So now the problem is that it is not feasible to have "grouped" features, such as x,y,r

The solution I am going to apply to this is taking the groups of 3 instances, and assigning them coordinates with a gridspace of 16, and then putting their rssi data at the relative location of the beacon, which of course means that I should not be using quadrants for the data


This is now complete, the time to classify is here.
Need multiclass(?), I am pretty sure


'''
